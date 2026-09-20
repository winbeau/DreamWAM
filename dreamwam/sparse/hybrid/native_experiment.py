"""Bounded, predeclared native-selector ablations and inclusive call budgets."""

from dataclasses import replace

from .config import HybridConfig
from .native_config import NativeRoutingConfig
from .schedule import Schedule


def uniform_feature_control(backend="cuda_graph"):
    # The inherited 56/294, [19,19,18] D0/R1-9 feature-reuse control.
    return HybridConfig(Schedule(10, ("dense",) + ("reuse",) * 9),
        recompute_ratio=0.1, read_mode="compact", read_ratio=0.1875,
        # Historical selector name is drift, but its Dense-anchor rule is
        # uniform and there are no Sparse steps to invoke a drift refresh.
        selection="drift", frame_quota="balanced", backend=backend)


def native_candidates(stage, backend="cuda_graph"):
    base = replace(uniform_feature_control(backend), read_ratio=1., read_count=56,
                   selection="native", native_routing=NativeRoutingConfig())
    rows = []

    def add(label, *, read_count=56, **options):
        native = NativeRoutingConfig(**options)
        rows.append((label, replace(base, read_count=read_count, native_routing=native)))

    if stage == "selectors":
        for signal in ("uniform", "action", "value_norm", "value_action", "dynamic", "visual_context", "action_context"):
            add(signal, signal=signal, observed="uniform" if signal == "dynamic" else "score")
        add("value_action_uniform_observed", signal="value_action", observed="uniform")
        # Calibration candidates, not a default mixture. Endpoints are the
        # value-action-uniform-observed/dynamic arms above, with identical frame
        # treatment. Freeze only after reviewing development outcomes.
        for weight in (0.25, 0.5, 0.75):
            add(f"value_action_dynamic_{weight:g}", signal="fusion", observed="uniform",
                weights=(("dynamic", weight), ("value_action", 1 - weight)))
    elif stage == "layers":
        for signal in ("value_action", "dynamic", "action_context"):
            add("layerwise_" + signal, signal=signal, layerwise=True,
                observed="uniform" if signal == "dynamic" else "score")
        add("first_layer_value_action", signal="value_action", layers=(0,))
    elif stage == "pooling":
        for signal in ("uniform", "dynamic"):
            add("hard198_" + signal, signal=signal, observed="full", read_count=198)
            add("pool198_" + signal, signal=signal, observed="full", read_count=198,
                packing="pool", refine_fraction=0.25)
        add("pool198_dynamic_unit", signal="dynamic", observed="full", read_count=198,
            packing="pool", refine_fraction=0.25, multiplicity="unit")
    elif stage == "refresh":
        # Exhaust all single Sparse positions; the no-refresh member anchors
        # this new cohort. Recompute urgency is independent of the AV/VV rank.
        context = replace(base, native_routing=NativeRoutingConfig(signal="action_context", recompute="drift"))
        rows.append(("context_no_refresh", context))
        for step in range(1, 10):
            operations = list(base.schedule.operations)
            operations[step] = "sparse"
            rows.append((f"context_sparse_{step}", replace(context, schedule=Schedule(10, tuple(operations)))))
    elif stage == "structure":
        for signal in ("uniform", "action_context"):
            rows.append(("structure_" + signal, replace(base, recompute_ratio=0.19,
                reuse_mode="structure", native_routing=NativeRoutingConfig(signal=signal))))
    else:
        raise ValueError("unknown bounded native stage")
    if len({config.policy_hash for _, config in rows}) != len(rows):
        raise ValueError("duplicate native candidates")
    return rows


def planned_calls(cells, input_count):
    """Include eager references, capture, prompt-miss checks and warm samples.

    One prediction per variant/input provides an eager reference; another
    captures its graph. One additional call/variant checks a warm graph with
    a cold prompt, and each group restores all prompt hits through Dense.
    Graph-internal warmups are transformer executions, not extra predictions.
    """
    groups = {row["group"] for row in cells}
    appearances = sum(len({row["variant"] for row in cells if row["group"] == group}) for group in groups)
    return dict(eager=appearances * input_count, capture=appearances * input_count,
                prompt_miss=appearances, restore_prompts=len(groups) * input_count,
                timed=len(cells), total=appearances * (2 * input_count + 1) + len(groups) * input_count + len(cells))
