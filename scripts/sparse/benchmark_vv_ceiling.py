#!/usr/bin/env python3
"""S1b: how much can VV sparsity actually save on DreamWAM's real shapes?

``profile_dense.py`` measures self-attention at ~5% of a complete request.  This script
turns that share into a decision: it benchmarks the *real* attention call shapes with the
kernel variants Sparse-WAM would use, including the gather cost, and reports the saving
both in milliseconds and as a fraction of a complete request.

Variants, all at B=1, H=24, D=128, Nv=294, P=98 (frame 0), Na=32:

* ``current``      the shipped fused call: [326] queries against [326] keys with a 2D mask
* ``split_dense``  video rows and action rows separated, no sparsity (S2 control)
* ``sparse``       frame-0 rows dense over 98 keys, remaining rows gathered to Kmax keys
* ``sparse_av``    as ``sparse`` plus the manual action branch that yields AV mass

``Kmax = P + keep_blocks * block_size``.  The implied request saving multiplies the
per-layer-step delta by 30 layers x 10 steps against the measured request latency, so a
result here is an *upper bound*: it assumes routing and scoring are free.

    .venv/bin/python scripts/sparse/benchmark_vv_ceiling.py --request-ms 514
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def time_call(function, *, warmup: int, iters: int) -> float:
    """Mean CUDA-event milliseconds for one call of ``function``."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def legal_mask(num_frames: int, tokens_per_frame: int, device) -> torch.Tensor:
    """Native DreamWAM video visibility: frame-0 queries see only frame-0 keys."""
    length = num_frames * tokens_per_frame
    frame_of = torch.arange(length, device=device) // tokens_per_frame
    return (frame_of != 0).unsqueeze(1) | (frame_of == 0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens-per-frame", type=int, default=98)
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--action-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=14)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--request-ms", type=float, default=514.0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    H, D = args.num_heads, args.head_dim
    P = args.tokens_per_frame
    Nv = args.num_frames * P
    Na = args.action_tokens
    width = H * D
    layer_steps = args.layers * args.steps

    generator = torch.Generator(device=device).manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    def rand_head(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * D**-0.5

    q_video, k_video, v_video = rand_head(Nv, width), rand_head(Nv, width), rand(Nv, width)
    q_action, k_action, v_action = rand_head(Na, width), rand_head(Na, width), rand(Na, width)
    mask = legal_mask(args.num_frames, P, device)

    def split_last(tensor):
        return tensor.view(tensor.shape[0], H, D).transpose(0, 1)

    def current():
        query = torch.cat([q_video, q_action], dim=0)
        key = torch.cat([k_video, k_action], dim=0)
        value = torch.cat([v_video, v_action], dim=0)
        return F.scaled_dot_product_attention(
            split_last(query).unsqueeze(0),
            split_last(key).unsqueeze(0),
            split_last(value).unsqueeze(0),
            attn_mask=torch.cat(
                [torch.cat([mask, torch.zeros(Nv, Na, dtype=torch.bool, device=device)], 1),
                 torch.ones(Na, Nv + Na, dtype=torch.bool, device=device)], 0),
        )

    def split_dense():
        video = F.scaled_dot_product_attention(
            split_last(q_video).unsqueeze(0),
            split_last(k_video).unsqueeze(0),
            split_last(v_video).unsqueeze(0),
            attn_mask=mask,
        )
        action = F.scaled_dot_product_attention(
            split_last(q_action).unsqueeze(0),
            split_last(torch.cat([k_video, k_action], 0)).unsqueeze(0),
            split_last(torch.cat([v_video, v_action], 0)).unsqueeze(0),
        )
        return video, action

    def make_sparse(keep_blocks: int, with_av: bool, gather: bool = True):
        """Frame-0 rows dense, other rows over P + keep_blocks future blocks."""
        index = torch.arange(P, device=device)
        if keep_blocks > 0:
            future = torch.arange(P, Nv, device=device)
            blocks = future[: (future.numel() // args.block_size) * args.block_size].view(
                -1, args.block_size
            )
            chosen = blocks[:keep_blocks].reshape(-1)
            index = torch.cat([index, chosen])
        kmax = index.numel()
        gather_index = index.view(1, 1, kmax, 1).expand(1, H, kmax, D)

        def sparse():
            video = F.scaled_dot_product_attention(
                split_last(q_video[:P]).unsqueeze(0),
                split_last(k_video[:P]).unsqueeze(0),
                split_last(v_video[:P]).unsqueeze(0),
            )
            if gather:
                key = split_last(k_video).unsqueeze(0).gather(2, gather_index)
                value = split_last(v_video).unsqueeze(0).gather(2, gather_index)
            else:
                key = split_last(k_video[:kmax]).unsqueeze(0)
                value = split_last(v_video[:kmax]).unsqueeze(0)
            rest = F.scaled_dot_product_attention(
                split_last(q_video[P:]).unsqueeze(0),
                key,
                value,
            )
            video = torch.cat([video, rest], dim=2)
            if not with_av:
                action = F.scaled_dot_product_attention(
                    split_last(q_action).unsqueeze(0),
                    split_last(torch.cat([k_video, k_action], 0)).unsqueeze(0),
                    split_last(torch.cat([v_video, v_action], 0)).unsqueeze(0),
                )
            else:
                action, _ = manual_action()
            return video, action

        return sparse, kmax

    def manual_action():
        query = split_last(q_action).unsqueeze(0) * D**-0.5
        video_logits = query @ split_last(k_video).unsqueeze(0).transpose(-1, -2)
        action_logits = query @ split_last(k_action).unsqueeze(0).transpose(-1, -2)
        logits = torch.cat([video_logits, action_logits], dim=-1)
        probs = torch.softmax(logits.float(), dim=-1).to(dtype)
        mass = probs[..., :Nv].sum(dim=-2)
        out = probs[..., :Nv] @ split_last(v_video).unsqueeze(0) + probs[
            ..., Nv:
        ] @ split_last(v_action).unsqueeze(0)
        return out, mass

    results = {
        "script": "scripts/sparse/benchmark_vv_ceiling.py",
        "stage": "S1b-vv-ceiling",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "shape": {
            "num_heads": H,
            "head_dim": D,
            "video_tokens": Nv,
            "first_frame_tokens": P,
            "action_tokens": Na,
            "block_size": args.block_size,
            "layer_steps": layer_steps,
        },
        "request_ms_reference": args.request_ms,
        "variants": [],
    }

    measured = {
        "current": time_call(current, warmup=args.warmup, iters=args.iters),
        "split_dense": time_call(split_dense, warmup=args.warmup, iters=args.iters),
        "sparse_no_av": time_call(
            make_sparse(2, with_av=False)[0], warmup=args.warmup, iters=args.iters
        ),
        "sparse_av": time_call(
            make_sparse(2, with_av=True)[0], warmup=args.warmup, iters=args.iters
        ),
    }

    baseline_ms = measured["current"] * layer_steps
    for name, per_call in measured.items():
        entry = {
            "variant": name,
            "ms_per_layer_step": per_call,
            "ms_per_request": per_call * layer_steps,
            "saving_ms_per_request": baseline_ms - per_call * layer_steps,
            "saving_fraction_of_request": (baseline_ms - per_call * layer_steps)
            / args.request_ms,
            "speedup_bound": args.request_ms
            / (args.request_ms - (baseline_ms - per_call * layer_steps)),
        }
        if name.startswith("sparse"):
            entry["kept_blocks"] = 2
            entry["kmax"] = P + 2 * args.block_size
        results["variants"].append(entry)

    # Sweep the number of kept future blocks for the gather backend.
    sweep = []
    max_blocks = (Nv - P) // args.block_size
    for keep in [0, 1, 2, 4, 7, max_blocks]:
        if keep > max_blocks:
            continue
        sparse, kmax = make_sparse(keep, with_av=True)
        per_call = time_call(sparse, warmup=args.warmup, iters=args.iters)
        saving = baseline_ms - per_call * layer_steps
        sweep.append(
            {
                "kept_blocks": keep,
                "kmax": kmax,
                "video_density": kmax / Nv,
                "ms_per_request": per_call * layer_steps,
                "saving_ms_per_request": saving,
                "saving_fraction_of_request": saving / args.request_ms,
                "speedup_bound": args.request_ms / (args.request_ms - saving),
            }
        )
    results["block_sweep"] = sweep

    # Isolate *why* the gathered variants lose. A contiguous key prefix needs no gather
    # and no index tensor, so if a prefix is fast while gather is slow, the cost is the
    # copy rather than the reduced key count; if both are slow, splitting the single
    # fused call is itself the cost and no mask policy can pay for it at this shape.
    prefix_sweep = []
    for keep in [0, 2, 4, 7]:
        kmax = P + keep * args.block_size
        if kmax > Nv:
            continue

        def prefix(kmax=kmax):
            video = F.scaled_dot_product_attention(
                split_last(q_video[:P]).unsqueeze(0),
                split_last(k_video[:P]).unsqueeze(0),
                split_last(v_video[:P]).unsqueeze(0),
            )
            rest = F.scaled_dot_product_attention(
                split_last(q_video[P:]).unsqueeze(0),
                split_last(k_video[:kmax]).unsqueeze(0),
                split_last(v_video[:kmax]).unsqueeze(0),
            )
            action, _ = manual_action()
            return torch.cat([video, rest], dim=2), action

        per_call = time_call(prefix, warmup=args.warmup, iters=args.iters)
        saving = baseline_ms - per_call * layer_steps
        prefix_sweep.append(
            {
                "kept_blocks": keep,
                "kmax": kmax,
                "video_density": kmax / Nv,
                "ms_per_request": per_call * layer_steps,
                "saving_ms_per_request": saving,
                "saving_fraction_of_request": saving / args.request_ms,
                "speedup_bound": args.request_ms / (args.request_ms - saving),
            }
        )
    results["contiguous_prefix_sweep"] = prefix_sweep

    # Cost of the gather itself, with no attention attached.
    gather_index = torch.arange(P, device=device).view(1, 1, P, 1).expand(1, H, P, D)
    key_head = split_last(k_video).unsqueeze(0)
    value_head = split_last(v_video).unsqueeze(0)
    results["gather_only_ms_per_layer_step"] = time_call(
        lambda: (
            key_head.gather(2, gather_index),
            value_head.gather(2, gather_index),
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    results["manual_action_ms_per_layer_step"] = time_call(
        manual_action, warmup=args.warmup, iters=args.iters
    )
    results["dense_video_only_ms_per_layer_step"] = time_call(
        lambda: F.scaled_dot_product_attention(
            split_last(q_video).unsqueeze(0),
            split_last(k_video).unsqueeze(0),
            split_last(v_video).unsqueeze(0),
            attn_mask=mask,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
