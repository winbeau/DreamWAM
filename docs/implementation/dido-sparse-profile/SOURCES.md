# Source and raw-data audit

Checked 2026-09-20 UTC from clean source `fda0235`. Read-only retrieval and
historical hash verification exited 0; no model run or training. The full
objective remains in [GOAL.md](GOAL.md).

## Paper and release status

[DIDO v2](https://arxiv.org/pdf/2609.15570v2), submitted 2026-09-15 05:54:35 UTC,
SHA-256 `9ab686c28188f03b580d80254153f4960fb72ab6ed16e6221c158bba609d8465`.
The PDF is 21 pages although its abstract-page comment says 22; retain the
downloaded version identity. Its arXiv license is non-exclusive distribution,
not a license to an unpublished dataset.

§3.3/Fig.3 rank future/current value differences, keep detailed high-change
regions and pool other K/V regions **for action reads**. Video computation and
other token groups are not pruned by that operation. It is a candidate inference
mechanism; effectiveness on untrained DreamWAM is an open experiment. DMD,
interaction tokens, box prediction and feature alignment require training.

Appendix A obtains object boxes with detection/tracking and simulated gripper
boxes through projection; those annotations are not original demonstrations.
Appendix D.2 defines group attribution share/enrichment but leaves the per-token
action V-attribution computation unspecified. Attention×value norms here are
our proxies. Token counts differ between Fig.3, Appendix B.2 and Fig.8; do not
infer a consistent released tensor layout from these figures. These statements
describe the paper, not implementation or validation in DreamWAM.

[Author repository](https://github.com/LoveJu1y/DIDO-WAM) HEAD
`56ea1652aea147ab3f7b9baffad2583885e5df57` (2026-09-15 04:17:42 UTC) contains
only README, blob `c35433896a43b66134a5391b42cd18386ead19bc`. No releases;
GitHub license field is null. README says code will come soon.
[Project page](https://loveju1y.github.io/DIDO/) exposes paper/figure PDFs and
eight demonstration **videos**, with no raw demonstration/annotation/attribution
download link observed. Videos and plotted results are not underlying raw records.
The user confirms they have no private author-data source. No author data acquired.

## Distinct data classes

| Class | Source/version/format | License and availability | Use here |
|---|---|---|---|
| Original demonstrations used by DIDO | Paper identifies LIBERO, RoboTwin and real Galbot collections; exact files/splits/checksums unreleased | Author-specific provenance unavailable | Not obtained; no claim to reproduce authors' samples |
| Author object/gripper tracks, boxes and DINO targets | Appendix A-derived records; schema and artifacts unreleased | No dataset license/link found | Not obtained; semantic coverage unknown |
| Author attribution/figure raw records | Appendix D.2 aggregate results; per-token definition and records absent | No artifact/license found | Not obtained; no exact V-attribution reproduction |
| Public LIBERO demonstrations | [Official download entrypoint](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets), suite HDF5 data | Official README declares dataset CC BY 4.0, code MIT; download path public; payload not downloaded or validated in this study | Possible separately labelled alternative, never equated to DIDO's exact training data |
| Our closed-loop observations | H100 `hybrid-routing-profile-20260920/trajectory-dense/manifest.json`, 9 NPZs, Spatial tasks 0/1/2 × init 1, first/middle/last | User-authorized local evaluation data; LIBERO source `8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT; no third-party data relicensing implied | Development only; hashes revalidated |
| Our native Q/K/V, hidden states and latents | To be captured from frozen DreamWAM checkpoint on the above inputs | Model-derived research artifacts, not demonstrations or author annotations | Pending raw capture; no future ground truth/privileged state enters policy |

The actual policy has two 224² center-cropped camera inputs concatenated across
width. VAE spatial downsampling is 16 and DiT patching is `(1,2,2)`: the expected
grid is `(3,7,14)` with two nominal 7×7 camera footprints, 98 cells per frame and
294 visual tokens. Runtime must assert the **actual** grid. Flattening follows
frame/row/column; camera is column//7, never contiguous groups of 49 in flattened
order. VAE receptive fields and subsequent attention can cross the image seam;
footprint identity is not exclusive semantic ownership. Original RoPE is retained.

## Retrieval artifacts and commands

Raw source bytes are archived outside Git at H100
`outputs/dido-sparse-profile-20260920/sources/`. [source-manifest.json](source-manifest.json)
records hashes and URLs. Retrieval used `curl -fsSL --max-time 30 URL -o FILE`;
GitHub `/commits/main`, `/contents/` and `/releases` were inspected in addition
to repository metadata and project links. No detector, dataset or dependency was
installed. Screenshot retrieval via the web PDF renderer failed; PDF bytes and
text were accessible. No failed retrieval is scientific evidence.
