"""Descriptive summaries of verified profiles, without treating tokens as trials."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .analysis import SCORES, jaccard, stamp, write_json
from .archive import sha256


def describe(values):
    return dict(count=len(values), mean=statistics.fmean(values), min=min(values), max=max(values))


def summarize_analysis(analysis_root, out_dir, *, plot=False):
    analysis_root, out_dir = Path(analysis_root), Path(out_dir)
    original = json.loads((analysis_root / "report.json").read_text())
    if original.get("status") != "COMPLETE" or original.get("exit_code") != 0:
        raise ValueError("summary requires independently verified complete analysis")
    for name, expected in original["artifacts"].items():
        if Path(name).name != name or sha256(analysis_root / name) != expected:
            raise ValueError("analysis artifact hash differs from verified report")
    out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=stamp(), argv=sys.argv,
        source_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3]),
            "rev-parse", "HEAD"], text=True).strip(),
        analysis_root=str(analysis_root.resolve()), analysis_report_sha256=sha256(analysis_root / "report.json"),
        input_kind=original["input_kind"], real_model_evidence=original["real_model_evidence"],
        capture_source_commit=original["source_commit"], checkpoint_sha256=original["checkpoint_sha256"],
        read_count=original["read_count"], split_role=original["split_role"], sr=None, speedup=None,
        limitations=["descriptive ranges over repeated observations, layers and heads; not independent trial confidence intervals",
            "support agreement does not establish causal importance or action fidelity",
            "video-time scores tie at zero in the current frame; future-only comparisons are also reported",
            "no semantic annotation, new rollout, learned fusion or chosen runtime policy"])
    write_json(out_dir / "report.json", report)
    try:
        curves = defaultdict(list)
        for line in (analysis_root / "stability.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["frame"] is None:
                curves[row["axis"], row["method"], row["after"]].append(row["jaccard"])
        for line in (analysis_root / "head-agreement.jsonl").read_text().splitlines():
            row = json.loads(line)
            curves["head", row["method"], row["step"]].append(row["jaccard"])
        routes = {}
        for line in (analysis_root / "selections.jsonl").read_text().splitlines():
            row = json.loads(line)
            routes[row["input_id"], row["step"], row["layer"], row["method"]] = row["indices"]
        steps, layers = (sorted(original["sampling"]["capture_" + key]) for key in ("steps", "layers"))
        # Frame/camera metadata comes from the validated raw-grid token table.
        regions = defaultdict(lambda: [0] * (1 + len(SCORES)))
        frame_tokens = defaultdict(set)
        with (analysis_root / "token-scores.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                name = row["input_id"]
                frame = int(row["frame"])
                frame_tokens[name, frame].add(int(row["token"]))
                key = (name, *(int(row[k]) for k in ("step", "layer", "head", "frame", "camera")))
                aggregate = regions[key]
                aggregate[0] += 1
                for i, score in enumerate(SCORES, start=1):
                    aggregate[i] += float(row[score])
        with (out_dir / "region-means.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["input_id", "step", "layer", "head", "frame", "camera", "tokens", *SCORES])
            for key, values in sorted(regions.items()):
                writer.writerow([*key, values[0], *(value / values[0] for value in values[1:])])
        cross_selectors, early_late = defaultdict(list), defaultdict(list)
        for (name, step, layer, method), route in routes.items():
            future = {token for (input_id, frame), tokens in frame_tokens.items()
                      if input_id == name and frame > 0 for token in tokens}
            early = routes[name, steps[0], layers[0], method]
            curves["early_reference_future", method, step].append(jaccard(set(route) & future, set(early) & future))
            if step == steps[-1] and layer == layers[-1]:
                early_late[method, "all"].append(jaccard(route, early))
                early_late[method, "future"].append(jaccard(set(route) & future, set(early) & future))
            if method in SCORES:
                for control in ("uniform", "action"):
                    other = routes[name, step, layer, control]
                    cross_selectors[method, control, "all"].append(jaccard(route, other))
                    cross_selectors[method, control, "future"].append(jaccard(set(route) & future, set(other) & future))
        summarized = [dict(axis=axis, method=method, after=position, **describe(values))
                      for (axis, method, position), values in sorted(curves.items())]
        write_json(out_dir / "curves.json", summarized)
        report.update(input_count=len({key[0] for key in routes}), sampling=original["sampling"],
            region_rows=len(regions),
            early_first_layer_to_final_last_layer=[dict(method=method, scope=scope, **describe(values))
                for (method, scope), values in sorted(early_late.items())],
            selector_overlap=[dict(method=method, control=control, scope=scope, **describe(values))
                for (method, control, scope), values in sorted(cross_selectors.items())])
        artifact_names = ["region-means.csv", "curves.json"]
        if plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            report["matplotlib_version"] = matplotlib.__version__
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            panels = [("denoising_step", "Adjacent denoising steps", "Later denoising step"),
                ("layer", "Consecutive sampled layers (0, 9, 19, 29)", "Later sampled layer"),
                ("head", "Pairwise sampled-head agreement", "Denoising step"),
                ("early_reference_future", "Future support vs step 0 / layer 0", "Denoising step")]
            for ax, (axis, title, xlabel) in zip(axes.flat, panels):
                for method in SCORES:
                    rows = [r for r in summarized if r["axis"] == axis and r["method"] == method]
                    ax.plot([r["after"] for r in rows], [r["mean"] for r in rows], marker=".", label=method)
                ax.set(title=title, xlabel=xlabel, ylabel="Mean support Jaccard", ylim=(-0.02, 1.02))
                ax.grid(alpha=0.2)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"DreamWAM self-captured development profile: {report['input_count']} observations, read {original['read_count']}\n"
                         "Descriptive agreement; no causal, speed or SR claim", fontsize=12)
            for name in ("support-agreement.png", "support-agreement.pdf"):
                fig.savefig(out_dir / name, dpi=170)
                artifact_names.append(name)
            plt.close(fig)
        report.update(status="COMPLETE", exit_code=0, artifacts={name: sha256(out_dir / name) for name in artifact_names})
    except BaseException as exc:
        report.update(status="ERROR", exit_code=1, error=repr(exc))
        raise
    finally:
        report["end_utc"] = stamp()
        write_json(out_dir / "report.json", report)
    return report
