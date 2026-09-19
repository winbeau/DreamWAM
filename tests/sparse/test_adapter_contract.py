"""Static contract checks on the action-eval adapter.

These catch a real, expensive failure: the adapter imports DreamWAM's entry points inside
``_import_dreamwam`` and publishes them through a dict, but a name that is imported without
being published raises ``NameError`` only when a worker starts on the evaluation server,
where it shows up as a crash loop that costs GPU reload cycles to diagnose. Parsing the file
needs no torch, no checkpoint and no GPU, so the mistake is catchable before deployment.
"""

from __future__ import annotations

import ast
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2] / "evaluation" / "action_eval" / "infer.py"


def _module() -> ast.Module:
    return ast.parse(ADAPTER.read_text())


def _published_names(tree: ast.Module) -> set[str]:
    published: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "_DREAMWAM"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    published.add(target.slice.value)
    return published


def _consumed_names(tree: ast.Module) -> set[str]:
    consumed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "dreamwam"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            consumed.add(node.slice.value)
    return consumed


def test_every_consumed_entry_point_is_published():
    tree = _module()
    published = _published_names(tree)
    consumed = _consumed_names(tree)
    missing = sorted(consumed - published)
    assert not missing, (
        "the adapter reads dreamwam[...] entries that _import_dreamwam never publishes: "
        f"{missing}. This is a NameError at worker startup, not at import time."
    )


def test_sparse_options_reach_build_policy():
    """The adapter must consume options['sparse']; a requested option is not an executed one."""
    source = ADAPTER.read_text()
    assert '"sparse"' in source or "'sparse'" in source, (
        "the adapter never reads policy.options['sparse'], so a sparse experiment YAML "
        "would silently execute dense"
    )
    assert "sparse=" in source, (
        "the adapter reads the sparse options but does not pass them to build_policy, so "
        "they would be ignored"
    )


def test_fingerprint_reports_the_sparse_config():
    source = ADAPTER.read_text()
    assert "sparse_config_hash" in source, (
        "the fingerprint must carry the effective sparse configuration, otherwise a run "
        "cannot be audited against what it claims to have executed"
    )
