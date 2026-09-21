"""Chinese reading aids from verified trace tensors; never alter source captures."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from observation_sequences import sha256
from trace_gallery import _attention_values, _save, token_mapping


def executed_values(record, arrays, length):
    """Absent keys are missing data, distinct from read keys with zero weight."""
    values = _attention_values(record, arrays, length, None, None)[0]
    read = np.zeros(length, dtype=bool)
    read[arrays[record["visual_key_ids"]]] = True
    return np.ma.array(values, mask=~read)


def vv_values(probe, arrays):
    if probe is None or "support" not in probe:
        return None
    values = arrays[probe["support"]].mean(0)
    # Older captures have no saved native visibility mask: do not infer one
    # from numerical zeros, which could also mean a very small probability.
    visible = arrays[probe["vv_visible_keys"]] if "vv_visible_keys" in probe else np.ones(len(values), bool)
    return np.ma.array(values, mask=~visible)


def _axes(ax, rows, cols, *, ids=False, offset=0):
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("patch 横坐标：0–6 外部相机；7–13 腕部相机", fontsize=9)
    ax.set_ylabel("patch 纵坐标", fontsize=9)
    ax.set_xticks(np.arange(cols + 1) - .5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", lw=.45, alpha=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(cols / 2 - .5, color="white", lw=2)
    if ids:
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, str(offset + r * cols + c), ha="center", va="center", color="white",
                        fontsize=7, bbox=dict(facecolor="black", alpha=.6, pad=.2, edgecolor="none"))


def _heat(ax, values, frame, grid, *, vmax, title, seeds=()):
    frames, rows, cols = grid
    ax.set_title(title, fontsize=11)
    if values is None:
        ax.set_facecolor("#eeeeee")
        ax.text(.5, .5, "本步没有运行路由探针\n没有热力值，不代表注意力为零", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return None
    matrix = np.ma.asarray(values).reshape(frames, rows, cols)[frame]
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#d8d8d8")
    im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, interpolation="nearest")
    for r, c in np.argwhere(np.ma.getmaskarray(matrix)):
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                              hatch="///", edgecolor="#999999", linewidth=0))
    for token in seeds:
        f, cell = divmod(int(token), rows * cols)
        if f == frame:
            r, c = divmod(cell, cols)
            ax.plot(c, r, "o", mfc="none", mec="#00ffff", ms=8, mew=1.1)
    _axes(ax, rows, cols)
    return im


def render_explainer(source, destination, *, font):
    source, destination, font = Path(source), Path(destination), Path(font)
    meta = json.loads((source / "trace.json").read_text())
    if sha256(source / "tensors.npz") != meta["tensors_sha256"]:
        raise ValueError("trace tensor hash mismatch")
    if not meta["actual_vae_input_verified"] or not meta["bitwise_uninstrumented"]:
        raise ValueError("require verified real-input and uninstrumented action parity")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False})
    destination.mkdir(parents=True, exist_ok=False)
    with np.load(source / "tensors.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive}
    grid = tuple(meta["grid"]); frames, rows, cols = grid
    length, plane = int(np.prod(grid)), rows * cols
    mapping = token_mapping(grid, arrays["model_input_rgb"].shape,
                           {k: arrays["raw_" + k].shape for k in ("agentview", "wrist")})
    (destination / "token-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    probes = {r["step"]: r for r in meta["records"] if r["kind"] == "router_probe"}
    work = {r["step"]: r for r in meta["records"] if r["kind"] == "executed_work"}
    gallery = []
    for record in [r for r in meta["records"] if r["kind"] == "executed_attention"]:
        step, layer = record["step"], record["layer"]
        probe = probes.get(step)
        av = _attention_values(probe, arrays, length, None, None)[0] if probe else None
        actual, vv = executed_values(record, arrays, length), vv_values(probe, arrays)
        seeds = arrays[probe["seeds"]] if probe and "seeds" in probe else ()
        query_ids = set(arrays[work[step]["query_ids"]].tolist())
        key_ids = set(arrays[record["visual_key_ids"]].tolist())
        av_max = max(float(actual.max()), float(av.max()) if av is not None else 0, 1e-12)
        vv_max = max(float(vv.max()), 1e-12) if vv is not None else 1
        title = f"{meta['episode_id']}  chunk {meta['call_index']}  |  去噪步 {step}：{record['operation']}"
        if layer == 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
            ax = axes[0, 0]
            ax.imshow(arrays["model_input_rgb"], extent=(-.5, cols - .5, rows - .5, -.5))
            _axes(ax, rows, cols, ids=True)
            ax.set_title("① 看原图：当前真实首帧 + token 编号", fontsize=12)
            im = _heat(axes[0, 1], av, 0, grid, vmax=av_max,
                       title="② 看 AV：动作直接关注哪些位置？", seeds=seeds)
            if im is not None: fig.colorbar(im, ax=axes[0, 1], shrink=.7, label="AV 概率")
            im = _heat(axes[1, 0], vv, 0, grid, vmax=vv_max,
                       title="③ 看 VV：圆圈种子关联哪些视觉位置？", seeds=seeds)
            if im is not None: fig.colorbar(im, ax=axes[1, 0], shrink=.7, label="VV 支持度（独立色标）")
            activity = np.array([1 if i in key_ids else 0 for i in range(plane)]).reshape(rows, cols)
            ax = axes[1, 1]
            ax.imshow(activity, cmap=ListedColormap(["#dedede", "#4682b4"]), vmin=0, vmax=1)
            for token in sorted(query_ids):
                if token < plane:
                    r, c = divmod(token, cols)
                    ax.plot(c, r, "+", color="#57ff57", ms=10, mew=1.6)
            _axes(ax, rows, cols)
            ax.set_title("④ 看实际执行：蓝色=读取；绿色 +=重算；灰色=未读", fontsize=11)
            fig.suptitle(title + "\n先按 ①→②→③→④ 阅读；四格都是同一张真实首帧，不是四个时间点。\n"
                "②③ 青色圆圈=AV 选出的种子；深色=数值低；灰色斜线=原始掩码禁止关联。\n"
                "AV 汇总所有注意力头与动作 token；VV 为加权种子支持度。注意力不是因果重要性。", fontsize=12)
            gallery.append(_save(fig, destination, f"s{step:02d}-read-this-first"))
        fig, axes = plt.subplots(frames, 3, figsize=(16, 3.9 * frames), constrained_layout=True)
        for frame in range(frames):
            row_label = ("第1行：当前真实首帧" if frame == 0 else f"第{frame+1}行：未来 latent {frame}（无真实 RGB）")
            for column, (value, scale, label) in enumerate(zip((av, actual, vv), (av_max, av_max, vv_max),
                ("第1列：路由 AV（第0层）", f"第2列：实际 AV（第{layer}层）", "第3列：VV 支持（第0层）"))):
                im = _heat(axes[frame, column], value, frame, grid, vmax=scale,
                           title=f"{label}\n{row_label} · ID {frame*plane}–{(frame+1)*plane-1}",
                           seeds=seeds if column != 1 else ())
                if column == 1:
                    for token in query_ids:
                        f, cell = divmod(token, plane)
                        if f == frame:
                            r, c = divmod(cell, cols)
                            axes[frame, column].plot(c, r, "+", color="#57ff57", ms=6)
                if im is not None: fig.colorbar(im, ax=axes[frame, column], shrink=.7, pad=.015)
        fig.suptitle(title + "\n横向三列=三种指标；纵向三行=三个视觉 latent 帧。同一坐标对应同一空间位置。\n"
            "第2列：灰色斜线=未读取，深色=读了但权重低，绿色 +=本步重算。第3列：灰色斜线=原始掩码禁止关联。\n"
            "青色圆圈=AV 种子；前两列共用 AV 色标，VV 单独色标。所有头/动作 token 汇总；数值由真实 Q/K 重建。", fontsize=12)
        gallery.append(_save(fig, destination, f"s{step:02d}-l{layer:02d}-rows-and-columns"))
    ordered = [work[k] for k in sorted(work)]
    refreshed = np.zeros((len(ordered), length), dtype=int)
    for index, record in enumerate(ordered): refreshed[index, arrays[record["query_ids"]]] = 1
    stats = meta["diagnostics"]["hybrid_visual"]
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.imshow(refreshed, aspect="auto", cmap=ListedColormap(["#eeeeee", "#2ca25f"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(ordered)), [f"步 {r['step']}：{r['operation']}" for r in ordered])
    for f in range(1, frames): ax.axvline(f * plane - .5, color="black", lw=1)
    ax.set_xlabel("视觉 token ID：0–97 当前首帧 | 98–195 未来 latent 1 | 196–293 未来 latent 2")
    ax.set_title(f"M1→M3 预算账本：实际重算 {stats['query_rows_spent']} 行 / chunk 上限 {stats.get('query_row_cap', '旧版本未记录')} 行\n"
                 "每行是一去噪步，每列是一 token；绿色=本步重算，浅灰=本步未重算。每步仍计算动作分支。")
    gallery.append(_save(fig, destination, "m1-m3-budget-ledger"))
    links = "\n".join(f'<figure><a href="{name}"><img src="{name}" loading="lazy"></a><figcaption>{name}</figcaption></figure>' for name in gallery)
    (destination / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>如何读 M1–M3 实验图</title>'
        '<style>body{font:18px system-ui;max-width:1600px;margin:2em auto}img{max-width:100%}figure{margin:2em 0}</style>'
        f'<h1>{html.escape(meta["episode_id"])} · chunk {meta["call_index"]}</h1>'
        '<p>先看 read-this-first 四格图，再看 rows-and-columns 九格图。大图的行列表示帧与指标；小格内部的坐标表示 patch 位置。</p>'
        '<p>首步 Dense 会读取并重算所有 token。未来 latent 没有输入真实 RGB，不能当作未来真实照片。'
        'patch 是名义空间格，VAE 感受野有重叠；热力图不是像素归因。</p>' + links, encoding="utf-8")
    report = dict(status="RENDERED_VERIFIED_TENSORS", source=str(source), source_commit=meta["commit"],
                  source_trace_sha256=sha256(source / "trace.json"), tensors_sha256=meta["tensors_sha256"],
                  font_sha256=sha256(font), images=gallery,
                  artifacts={p.name: sha256(p) for p in sorted(destination.iterdir()) if p.is_file()})
    report["quick_start"] = next((f"s{r['step']:02d}-read-this-first.png" for r in ordered
        if r["operation"] == "sparse" and f"s{r['step']:02d}-read-this-first.png" in gallery), gallery[0])
    (destination / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    sources = [args.source] if (args.source / "trace.json").is_file() else sorted(p.parent for p in args.source.glob("*/trace.json"))
    if not sources: raise ValueError("no trace captures found")
    reports = []
    for source in sources:
        report = render_explainer(source, args.out_dir / source.name, font=args.font)
        reports.append((source.name, report))
        print(json.dumps(dict(capture=source.name, status=report["status"], pngs=len(report["images"]))), flush=True)
    links = "\n".join(f'<li>{name}：<a href="{name}/{report["quick_start"]}">先看四格导读</a>'
        f' · <a href="{name}/index.html">全部图和行列解释</a></li>' for name, report in reports)
    (args.out_dir / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>M1–M3 中文读图入口</title>'
        '<style>body{font:20px system-ui;max-width:1100px;margin:2em auto;line-height:1.8}</style>'
        '<h1>M1–M3：从原图读到实际计算</h1><p>每个链接对应一个真实 chunk。优先打开实际 Sparse 步的四格导读。</p>'
        '<p>① 原图和 token 编号 → ② AV 注意力 → ③ VV 支持 → ④ 实际读/算。'
        '九格图横向三列是指标，纵向三行是真实首帧和两个未来 latent。</p>'
        '<p>绿色 +=本步重算；蓝色=读取；灰色=未读。注意力图中的灰色斜线表示缺失或被原始掩码禁止，'
        '深色才是有效位置的低数值。大块浅灰文字框表示本步没有运行探针。</p><ul>' + links + '</ul>', encoding="utf-8")
