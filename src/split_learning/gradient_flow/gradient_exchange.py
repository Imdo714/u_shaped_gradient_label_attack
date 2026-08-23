from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..architecture.split_learning_model import SplitLearningModel


COMMUNICATION_DIAGRAM = r"""
U자형 분할 학습 통신 구조
순전파: x -> ClientFront f -> z --전송--> ServerMiddle g -> u --전송--> ClientTail h -> logits -> CE(logits, y)
역전파: loss -> ClientTail h -> dL/du --전송--> ServerMiddle g -> dL/dz --전송--> ClientFront f
레이블 y는 ClientTail에만 존재하며 ServerMiddle은 전달받지 않습니다.
""".strip()


@dataclass
class GradientExchangeResult:
    """한 번의 명시적 분할 학습 단계에서 생성된 결과와 통신 텐서."""

    loss: float
    logits: Tensor
    smashed_z: Tensor
    server_output_u: Tensor
    grad_h_to_g: Tensor
    grad_g_to_f: Tensor


def _print_tensor_summary(name: str, tensor: Tensor, max_values: int) -> None:
    """큰 텐서를 전부 출력하지 않고 shape, 통계와 앞부분 값만 표시한다."""
    detached = tensor.detach()
    flat = detached.flatten()
    values = flat[:max_values].cpu().tolist()
    stats = detached.float()
    print(
        f"  {name:18s} shape={str(tuple(detached.shape)):18s} "
        f"requires_grad={str(tensor.requires_grad):5s} "
        f"min={stats.min().item(): .6f} max={stats.max().item(): .6f} "
        f"mean={stats.mean().item(): .6f} norm={stats.norm().item(): .6f}"
    )
    print(f"  {'':18s} first_{len(values)}={values}")


def _parameter_gradient_norm(module: nn.Module) -> float:
    """한 모델에 속한 모든 파라미터 gradient의 전체 L2 norm을 계산한다."""
    squared_norm = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm = squared_norm + parameter.grad.detach().float().pow(2).sum()
    return float(squared_norm.sqrt().cpu())


def run_gradient_exchange_step(
    model: SplitLearningModel,
    x: Tensor,
    client_labels: Tensor,
    criterion: nn.Module,
    optimizer_f: torch.optim.Optimizer | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_h: torch.optim.Optimizer | None = None,
    update: bool = True,
    debug: bool = False,
    debug_max_values: int = 8,
    debug_sample_ids: list[str] | tuple[str, ...] | None = None,
    debug_epoch: int | None = None,
    debug_batch_id: int | None = None,
) -> GradientExchangeResult:
    """두 분할 경계를 통과하는 순전파와 역전파 통신을 명시적으로 수행한다.

    `client_labels`는 두 번째 detach 이후 ClientTail에서만 사용한다.
    ServerMiddle이나 서버 transcript logger에는 레이블을 전달하지 않는다.
    """
    # 파라미터를 업데이트하려면 f, g, h 각각의 optimizer가 모두 필요하다.
    if update and any(opt is None for opt in (optimizer_f, optimizer_g, optimizer_h)):
        raise ValueError("All three optimizers are required when update=True")

    # 이전 학습 단계에서 남은 f, g, h의 파라미터 그래디언트를 초기화한다.
    for opt in (optimizer_f, optimizer_g, optimizer_h):
        if opt is not None:
            opt.zero_grad(set_to_none=True)

    # [순전파 1: 클라이언트 f]
    # ClientFront가 원본 이미지 x로부터 첫 번째 smashed data z를 계산한다.
    # z_local = 클라이언트 내부에서 계산된 smashed data
    z_local = model.f_model(x)

    # [통신 경계 1: ClientFront -> ServerMiddle]
    # detach()는 z가 직렬화되어 네트워크로 전달되면서 기존 autograd 그래프가
    # 끊기는 상황을 표현한다. 서버는 새 leaf tensor인 z_wire를 받는다.
    # z_wire  = 서버로 전송된 smashed data의 복사본
    z_wire = z_local.detach().requires_grad_(True)

    # [순전파 2: 서버 g]
    # 서버는 z만 관찰하고 ServerMiddle을 실행해 u를 계산한다.
    # 이 호출에는 client_labels가 전달되지 않는다.
    u_server = model.g_model(z_wire)

    # [통신 경계 2: ServerMiddle -> ClientTail]
    # 서버 출력 u를 ClientTail로 전송한다. 두 번째 detach로 서버와
    # ClientTail의 autograd 그래프도 명시적으로 분리한다.
    u_wire = u_server.detach().requires_grad_(True)

    # [순전파 3: 클라이언트 h]
    # 실제 레이블과 손실 함수는 ClientTail 쪽에만 존재한다.
    logits = model.h_model(u_wire)
    loss = criterion(logits, client_labels)

    # [역전파 1: ClientTail h]
    # detach 경계 때문에 이 backward는 ClientTail 파라미터와 u_wire까지만
    # 역전파한다. ServerMiddle과 ClientFront까지 자동으로 넘어가지 않는다.
    loss.backward()

    # 이것이 공격 대상 그래디언트 dL/du이다.
    # 손실 L이 실제 레이블 y를 사용하므로 ClientTail에서 생성된다.
    # ClientTail -> ServerMiddle로 전송되며, 서버는 이 텐서를 관찰하지만
    # 실제 레이블 y 자체는 전달받지 않는다.
    grad_h_to_g = u_wire.grad.detach().clone()

    # [역전파 2: ServerMiddle g]
    # 서버는 ClientTail에서 받은 dL/du를 외부 그래디언트로 사용해
    # ServerMiddle을 역전파한다. 이 과정에서 g의 파라미터 gradient와
    # 입력 z에 대한 dL/dz가 계산된다.
    u_server.backward(grad_h_to_g)

    # 이것이 서버에서 생성되어 ClientFront로 전송되는 dL/dz이다.
    grad_g_to_f = z_wire.grad.detach().clone()

    # [역전파 3: ClientFront f]
    # ClientFront는 서버가 보내준 dL/dz를 사용해 자신의 그래프를 역전파한다.
    z_local.backward(grad_g_to_f)

    # 최초 몇 개 샘플에서 x -> z -> u -> logits -> gradient 흐름을 출력한다.
    # 이 화면은 실제 레이블도 표시하는 연구자 디버그 view이다. 서버 공격자
    # transcript에는 이 레이블과 확률이 기록되지 않는다.
    if debug:
        probabilities = torch.softmax(logits.detach(), dim=1)
        predicted = logits.detach().argmax(dim=1)
        sample_text = ", ".join(debug_sample_ids or ["unknown"])
        print("\n" + "=" * 92)
        print("[연구자 디버그 view] 실제 샘플의 U자형 분할 학습 중간값 및 gradient")
        print("  주의: 실제 레이블은 설명을 위한 이 터미널에만 표시되며 서버 transcript에는 없음")
        print(
            f"  epoch={debug_epoch} batch_id={debug_batch_id} "
            f"sample_id={sample_text}"
        )

        print("\n[순전파 0] 클라이언트 입력")
        _print_tensor_summary("x (normalized input)", x, debug_max_values)

        print("\n[순전파 1] ClientFront f: x -> z")
        _print_tensor_summary("z_local", z_local, debug_max_values)
        print("  z_wire는 z_local을 detach하여 서버가 받은 smashed data를 표현")
        _print_tensor_summary("z_wire", z_wire, debug_max_values)

        print("\n[순전파 2] ServerMiddle g: z -> u")
        _print_tensor_summary("u_server", u_server, debug_max_values)
        print("  u_wire는 u_server를 detach하여 ClientTail이 받은 서버 출력을 표현")
        _print_tensor_summary("u_wire", u_wire, debug_max_values)

        print("\n[순전파 3] ClientTail h: u -> logits")
        _print_tensor_summary("logits", logits, debug_max_values)
        _print_tensor_summary("softmax probability", probabilities, debug_max_values)
        print(f"  client true label = {client_labels.detach().cpu().tolist()}")
        print(f"  predicted label   = {predicted.cpu().tolist()}")
        print(f"  cross entropy loss= {loss.detach().item():.8f}")

        print("\n[역전파 1] ClientTail h -> ServerMiddle g")
        print("  grad_h_to_g = dL/du: 레이블 영향을 받아 서버가 관찰하는 공격 대상")
        _print_tensor_summary("grad_h_to_g", grad_h_to_g, debug_max_values)

        print("\n[역전파 2] ServerMiddle g -> ClientFront f")
        print("  grad_g_to_f = dL/dz: 서버가 계산해 ClientFront로 보내는 gradient")
        _print_tensor_summary("grad_g_to_f", grad_g_to_f, debug_max_values)

        print("\n[파라미터 gradient 전체 L2 norm]")
        print(f"  ClientFront f  : {_parameter_gradient_norm(model.f_model):.8f}")
        print(f"  ServerMiddle g : {_parameter_gradient_norm(model.g_model):.8f}")
        print(f"  ClientTail h   : {_parameter_gradient_norm(model.h_model):.8f}")
        print("=" * 92)

    # 모든 그래디언트 계산이 끝난 뒤 h, g, f를 각각의 optimizer로 업데이트한다.
    if update:
        optimizer_h.step()  # type: ignore[union-attr]
        optimizer_g.step()  # type: ignore[union-attr]
        optimizer_f.step()  # type: ignore[union-attr]

    return GradientExchangeResult(
        loss=float(loss.detach()),
        logits=logits.detach(),
        smashed_z=z_wire.detach(),
        server_output_u=u_wire.detach(),
        grad_h_to_g=grad_h_to_g,
        grad_g_to_f=grad_g_to_f,
    )


def observe_frozen_gradient_exchange(
    model: SplitLearningModel,
    x: Tensor,
    client_labels: Tensor,
    criterion: nn.Module,
) -> GradientExchangeResult:
    """고정된 체크포인트에서 파라미터 업데이트 없이 학습 손실 그래디언트를 관찰한다."""

    # Dropout과 BatchNorm 상태를 고정하기 위해 평가 모드로 전환한다.
    model.eval()

    # 순전파 통신은 훈련 단계와 동일하게 f -> z -> g -> u -> h 순서로 수행한다.
    # z와 u의 detach는 두 통신 경계를 그대로 재현한다.
    z = model.f_model(x).detach().requires_grad_(True)
    u_server = model.g_model(z)
    u = u_server.detach().requires_grad_(True)

    # 레이블은 ClientTail에서만 사용해 교차 엔트로피 손실을 계산한다.
    logits = model.h_model(u)
    loss = criterion(logits, client_labels)

    # 공격자가 관찰하는 dL/du를 먼저 계산한다.
    grad_u = torch.autograd.grad(loss, u)[0]

    # 계산된 dL/du를 ServerMiddle의 외부 그래디언트로 전달해 dL/dz를 계산한다.
    # autograd.grad를 사용하므로 모델 파라미터를 업데이트하거나 optimizer를 실행하지 않는다.
    grad_z = torch.autograd.grad(u_server, z, grad_outputs=grad_u)[0]
    return GradientExchangeResult(
        float(loss.detach()), logits.detach(), z.detach(), u.detach(), grad_u.detach(), grad_z.detach()
    )


# Compatibility aliases for code using names from the previous flat package.
StepResult = GradientExchangeResult
explicit_split_step = run_gradient_exchange_step
frozen_gradient_observation = observe_frozen_gradient_exchange

__all__ = [
    "COMMUNICATION_DIAGRAM",
    "GradientExchangeResult",
    "run_gradient_exchange_step",
    "observe_frozen_gradient_exchange",
    "StepResult",
    "explicit_split_step",
    "frozen_gradient_observation",
]
