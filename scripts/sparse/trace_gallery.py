"""Render saved execution evidence; no model calls or fabricated attention."""

import csv
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from observation_sequences import sha256


def token_mapping(grid, image_shape, raw_shapes):
    frames, rows, cols = map(int, grid)
    height, width, channels = image_shape
    if min(frames, rows, cols) < 1 or channels != 3 or width != 2 * height or cols % 2:
        raise ValueError("expected a positive token grid aligned to two square camera inputs")
    result = []
    for token in range(frames * rows * cols):
        frame, spatial = divmod(token, rows * cols)
        row, col = divmod(spatial, cols)
        camera = "agentview" if col < cols // 2 else "wrist"
        x0, x1 = col * width / cols, (col + 1) * width / cols
        y0, y1 = row * height / rows, (row + 1) * height / rows
        offset = 0 if camera == "agentview" else height
        raw_h, raw_w, _ = raw_shapes[camera]
        scale = max(height / raw_w, height / raw_h)
        resized_w, resized_h = round(raw_w * scale), round(raw_h * scale)
        left, top = max((resized_w - height) // 2, 0), max((resized_h - height) // 2, 0)
        bbox = [(x0 - offset + left) * raw_w / resized_w,
                (y0 + top) * raw_h / resized_h,
                (x1 - offset + left) * raw_w / resized_w,
                (y1 + top) * raw_h / resized_h]
        result.append(dict(token_id=token, latent_frame=frame, row=row, col=col,
            camera=camera, observed_rgb=frame == 0, model_input_bbox=[x0, y0, x1, y1],
            raw_observation_bbox=bbox if frame == 0 else None,
            interpretation="nominal spatial cell; VAE receptive fields overlap"))
    return result


def _save(fig, directory, stem):
    for extension in ("png", "pdf"):
        fig.savefig(directory / (stem + "." + extension), dpi=155, bbox_inches="tight")
    plt.close(fig)
    return stem + ".png"


def _grid(ax, image, rows, cols, ids=False):
    height, width = image.shape[:2]
    ax.imshow(image)
    for row in range(rows + 1):
        ax.axhline(row * height / rows - .5, lw=.5, color="white", alpha=.8)
    for col in range(cols + 1):
        ax.axvline(col * width / cols - .5, lw=.5, color="white", alpha=.8)
    if ids:
        for row in range(rows):
            for col in range(cols):
                ax.text((col + .5) * width / cols, (row + .5) * height / rows,
                        str(row * cols + col), ha="center", va="center", fontsize=7,
                        color="white", bbox=dict(facecolor="black", alpha=.5, pad=.3, edgecolor="none"))
    ax.set_xticks([]); ax.set_yticks([])


def _attention_values(record, arrays, length, head, query):
    p = arrays[record["joint_probabilities"]][0]
    selected = p.mean(axis=(0, 1)) if head is None else p[head, query]
    keys = arrays[record["visual_key_ids"]] if "visual_key_ids" in record else np.arange(length)
    values = np.zeros(length, dtype=np.float32)
    values[keys] = selected[:len(keys)]
    return values, float(selected[:len(keys)].sum()), float(selected[len(keys):].sum())


def render_capture(directory, *, head=0, query=0):
    directory = Path(directory)
    meta = json.loads((directory / "trace.json").read_text())
    if sha256(directory / "tensors.npz") != meta["tensors_sha256"]:
        raise ValueError("trace tensor hash mismatch")
    with np.load(directory / "tensors.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive}
    frames, rows, cols = meta["grid"]
    length = frames * rows * cols
    pixels = arrays["model_input_rgb"]
    mapping = token_mapping(meta["grid"], pixels.shape,
                            {name: arrays["raw_" + name].shape for name in ("agentview", "wrist")})
    (directory / "token-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    title = f'{meta["episode_id"]} | chunk {meta["call_index"]} | Q/KV {meta["budget_label"]}'
    gallery = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, camera in zip(axes[0], ("agentview", "wrist")):
        ax.imshow(arrays["raw_" + camera]); ax.set_title("Raw observation: " + camera); ax.axis("off")
    axes[1, 0].imshow(pixels); axes[1, 0].set_title("Actual model crop/resize: agentview | wrist")
    axes[1, 0].axis("off")
    _grid(axes[1, 1], pixels, rows, cols, ids=True)
    axes[1, 1].set_title(f"Observed latent frame 0: IDs 0–{rows * cols - 1}")
    fig.suptitle(title + "\nGrid inferred from real pre_dit shape; boxes are nominal cells, not isolated receptive fields", fontsize=11)
    gallery.append(_save(fig, directory, "inputs-and-token-ids"))
    records = meta["records"]
    probes = {r["step"]: r for r in records if r["kind"] == "router_probe"}
    work = {r["step"]: r for r in records if r["kind"] == "executed_work"}
    executed = [r for r in records if r["kind"] == "executed_attention"]
    for record in executed:
        step, layer = record["step"], record["layer"]
        probe = probes.get(step)
        for chosen_head, chosen_query, label in ((None, None, "mean-heads-queries"), (head, query, f"h{head}-q{query}")):
            actual, av_mass, aa_mass = _attention_values(record, arrays, length, chosen_head, chosen_query)
            av = _attention_values(probe, arrays, length, chosen_head, chosen_query)[0] if probe else None
            vv = arrays[probe["support"]].mean(0) if probe and "support" in probe else None
            values = (av, actual, vv)
            vmax = max(float(actual.max()), float(av.max()) if av is not None else 0, 1e-12)
            fig, axes = plt.subplots(frames, 3, figsize=(15, 3.3 * frames), squeeze=False, constrained_layout=True)
            for column, (name, value) in enumerate(zip(("Router probe AV (layer 0)", f"Executed AV (layer {layer})", "VV backward support (layer 0)"), values)):
                for frame in range(frames):
                    ax = axes[frame, column]
                    ax.set_title(name + (" | observed frame 0" if frame == 0 else f" | future latent {frame}; no observed RGB"), fontsize=9)
                    if value is None:
                        ax.text(.5, .5, "No probe at this step", ha="center", va="center", transform=ax.transAxes)
                        ax.axis("off"); continue
                    matrix = value.reshape(frames, rows, cols)[frame]
                    if frame == 0:
                        ax.imshow(pixels, extent=(-.5, cols - .5, rows - .5, -.5))
                    im = ax.imshow(matrix, cmap="magma", vmin=0,
                                   vmax=vmax if column < 2 else max(float(value.max()), 1e-12),
                                   alpha=.78 if frame == 0 else 1)
                    # Key read set and query update set come from executed records.
                    if column == 1:
                        key_ids = set(arrays[record["visual_key_ids"]].tolist())
                        query_ids = set(arrays[work[step]["query_ids"]].tolist())
                        for spatial in range(rows * cols):
                            token = frame * rows * cols + spatial
                            r, c = divmod(spatial, cols)
                            if token in key_ids:
                                ax.add_patch(Rectangle((c - .47, r - .47), .94, .94, fill=False, ec="#00cfff", lw=.65))
                            if token in query_ids:
                                ax.plot(c, r, "+", color="#44ff55", markersize=5, markeredgewidth=.8)
                    elif probe and "seeds" in probe:
                        for token in arrays[probe["seeds"]]:
                            f, s = divmod(int(token), rows * cols)
                            if f == frame:
                                r, c = divmod(s, cols)
                                ax.plot(c, r, "o", mfc="none", mec="#00ffff", ms=5, mew=.9)
                    ax.set_xticks(range(cols)); ax.set_yticks(range(rows)); ax.tick_params(labelsize=6)
                    ax.set_xlabel("grid column", fontsize=8); ax.set_ylabel("grid row", fontsize=8)
                    fig.colorbar(im, ax=ax, shrink=.75, pad=.015)
            reduction = "mean over all heads and action queries" if chosen_head is None else f"head {head}, action query {query}"
            fig.suptitle(title + f" | denoising step {step} | {record['operation']}\n{reduction}; joint [V,A] softmax, no video renormalization; AV mass {av_mass:.3f}, AA mass {aa_mass:.3f}\n"
                + "Cyan box: executed visual key; green +: updated query; cyan circle: AV seed. VV: weighted seeds, mean heads.\n"
                + "Q/K reconstruction from actual native call; fused-kernel internal probabilities are not exposed.", fontsize=10)
            gallery.append(_save(fig, directory, f"s{step:02d}-l{layer:02d}-{label}"))
    # Ranked observed-frame crops and table retain exact token IDs and original units.
    if probes:
        first = probes[min(probes)]
        direct = arrays[first["direct"]].mean(0)
        support = arrays[first["support"]].mean(0) if "support" in first else np.zeros(length)
        read = arrays[first["read_score"]]
        ranked = np.argsort(-read[:rows * cols], kind="stable")[:8]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        table = []
        for ax, token in zip(axes.flat, ranked):
            item = mapping[int(token)]
            x0, y0, x1, y1 = item["model_input_bbox"]
            ax.imshow(pixels[math.floor(y0):math.ceil(y1), math.floor(x0):math.ceil(x1)])
            ax.axis("off")
            ax.set_title(f"ID {token} | {item['camera']} r{item['row']} c{item['col']}\nAV sum {direct[token]:.5f}; VV {support[token]:.5f}\nread score {read[token]:.4f}", fontsize=9)
            table.append(dict(token_id=int(token), camera=item["camera"], row=item["row"], col=item["col"],
                              av_sum_over_queries_mean_heads=float(direct[token]),
                              vv_support=float(support[token]), read_score=float(read[token])))
        fig.suptitle(title + f"\nTop observed cells by router read score at step {first['step']}; crops of actual model input", fontsize=11)
        gallery.append(_save(fig, directory, "observed-key-crops"))
        with (directory / "observed-key-scores.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    ordered = [work[step] for step in sorted(work)]
    age = np.stack([arrays[r["feature_ages"]] for r in ordered])
    updated = np.zeros_like(age)
    retained = np.zeros_like(age)
    for i, record in enumerate(ordered):
        updated[i, arrays[record["query_ids"]]] = 1
        retained[i, arrays[record["retained_key_ids"]]] = 1
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    for ax, matrix, name in zip(axes, (updated, retained, age),
                              ("Actually updated visual queries", "Retained visual keys after step", "Visual feature age (steps since recomputation)")):
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0)
        for frame in range(1, frames): ax.axvline(frame * rows * cols - .5, color="white", lw=1)
        ax.set_yticks(range(len(ordered)), [f"{r['step']}: {r['operation']}" for r in ordered], fontsize=8)
        ax.set_xlabel("Visual token ID; vertical lines separate latent frames"); ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(title + "\nEvery step updates the action branch; all visual cache state resets at the next chunk", fontsize=11)
    gallery.append(_save(fig, directory, "execution-and-cache-age"))
    links = "\n".join(f'<figure><a href="{name}"><img loading="lazy" src="{name}"></a><figcaption>{html.escape(name)} · <a href="{name[:-4]}.pdf">PDF</a></figcaption></figure>' for name in gallery)
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>DreamWAM execution trace</title>'
        '<style>body{font:16px system-ui;margin:2em auto;max-width:1500px}img{max-width:100%}figure{margin:2em 0}code{overflow-wrap:anywhere}</style>'
        f'<h1>{html.escape(title)}</h1><p>Real checkpoint execution. Instrumented eager capture; not a latency measurement. '
        f'Actions match the uninstrumented replay bitwise. Source <code>{meta["commit"]}</code>.</p>'
        '<p><a href="trace.json">Metadata</a> · <a href="tensors.npz">Raw tensors</a> · <a href="token-map.json">Token map</a>'
        ' · <a href="observed-key-scores.csv">Observed token scores</a>. Future latent grids have no ground-truth RGB. '
        'Colors display unrenormalized joint AV probabilities; VV support has a separate scale. Patch boxes denote spatial coordinates, not pixel attribution.</p>' + links)
    return {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
