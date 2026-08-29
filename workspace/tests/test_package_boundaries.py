from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


def imported_modules(package: str) -> set[str]:
    modules: set[str] = set()
    for path in (SOURCE_ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    return modules


def test_src_contains_only_the_runtime_packages():
    packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == {
        "split_learning",
        "shared",
        "experiments",
        "decoder",
        "transcript_inversion",
    }


def test_non_source_files_are_grouped_by_role():
    assert (WORKSPACE_ROOT / "data" / "dataset").is_dir()
    assert (WORKSPACE_ROOT / "data" / "anchors").is_dir()
    assert (WORKSPACE_ROOT / "results" / "checkpoints").is_dir()
    assert (WORKSPACE_ROOT / "results" / "transcripts").is_dir()
    assert (WORKSPACE_ROOT / "results" / "reports").is_dir()
    assert (WORKSPACE_ROOT / "results" / "runs").is_dir()
    assert (WORKSPACE_ROOT / "tests").is_dir()

    legacy_roots = ("data", "anchors", "checkpoints", "transcripts", "results", "runs", "tests")
    assert not any((PROJECT_ROOT / name).exists() for name in legacy_roots)


def test_dependency_direction_respects_package_responsibilities():
    shared_imports = imported_modules("shared")
    split_learning_imports = imported_modules("split_learning")
    decoder_imports = imported_modules("decoder")
    transcript_inversion_imports = imported_modules("transcript_inversion")

    assert not any(module.startswith("src.split_learning") for module in shared_imports)
    assert not any(module.startswith("src.experiments") for module in shared_imports)
    assert not any(module.startswith("src.experiments") for module in split_learning_imports)
    assert not any(module.startswith("src.experiments") for module in decoder_imports)
    assert not any(
        module.startswith("src.experiments") for module in transcript_inversion_imports
    )
