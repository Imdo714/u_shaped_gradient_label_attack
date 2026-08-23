import torch
from torch import nn

from src.split_learning.architecture.split_learning_model import (
    build_split_learning_model,
)
from src.split_learning.gradient_flow.gradient_exchange import (
    run_gradient_exchange_step,
)


def test_explicit_gradient_flow_updates_all_three_models():
    torch.manual_seed(3)
    model = build_split_learning_model("middle")
    criterion = nn.CrossEntropyLoss()
    optimizers = [
        torch.optim.SGD(model.f_model.parameters(), lr=0.05),
        torch.optim.SGD(model.g_model.parameters(), lr=0.05),
        torch.optim.SGD(model.h_model.parameters(), lr=0.05),
    ]
    components = [model.f_model, model.g_model, model.h_model]
    before = [[parameter.detach().clone() for parameter in component.parameters()] for component in components]
    result = run_gradient_exchange_step(
        model,
        torch.randn(2, 3, 32, 32),
        torch.tensor([0, 1]),
        criterion,
        *optimizers,
    )
    assert result.grad_h_to_g.shape == result.server_output_u.shape
    assert result.grad_g_to_f.shape == result.smashed_z.shape
    assert result.grad_h_to_g.norm() > 0
    assert result.grad_g_to_f.norm() > 0
    for old_parameters, component in zip(before, components):
        assert any(not torch.equal(old, new) for old, new in zip(old_parameters, component.parameters()))


def test_configurable_cut_shapes():
    expected = {
        "early": ((1, 16, 16, 16), (1, 64, 4, 4)),
        "middle": ((1, 32, 8, 8), (1, 64, 4, 4)),
        "late": ((1, 64, 4, 4), (1, 64, 4, 4)),
    }
    for cut, (z_shape, u_shape) in expected.items():
        model = build_split_learning_model(cut)
        z = model.f_model(torch.randn(1, 3, 32, 32))
        u = model.g_model(z)
        assert tuple(z.shape) == z_shape
        assert tuple(u.shape) == u_shape
