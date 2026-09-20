> **Evaluation experiment — `experiment/dido-sparse-profile`**: [active goal](docs/implementation/dido-sparse-profile/PROGRESS.md), [verified native profile and 47 bounded interventions](docs/implementation/dido-sparse-profile/RESULTS.md), [400-call online screen](docs/implementation/dido-sparse-profile/ONLINE-RESULTS.md), [online implementation](docs/implementation/dido-sparse-profile/ONLINE.md), [data sources](docs/implementation/dido-sparse-profile/SOURCES.md), [deployment workflow](docs/implementation/dido-sparse-profile/WORKFLOW.md). CPU/CUDA gates and the bounded D/R screen pass. Two retained selectors improve development action proxies at about 2× warm prediction speed; refresh/structure, final adapter and SR gates remain open. This effort is capped at 50 closed-loop episode attempts total across arms (currently 0), with a 5-percentage-point SR-drop tolerance. Historical [paper-evaluation results](docs/action-eval/README.md) remain separate. Upstream instructions and attribution below are preserved; execution remains server-only with unchanged dependencies.

<div align="center">

<h2>DreamWAM: Beyond RGB Future Prediction<br>for World Action Models</h2>

<p>
  <b>Shanglin Yuan</b><sup>1,2,*</sup> &middot;
  <b>Weiheng Zhao</b><sup>1,2,*</sup> &middot;
  <b>Xin Shi</b><sup>2</sup> &middot;
  <b>Haoyi Jiang</b><sup>1,2</sup> &middot;
  <b>Xianda Guo</b><sup>3</sup> &middot;
  <b>Liu Liu</b><sup>4</sup> &middot;
  <b>Wenyu Liu</b><sup>1</sup> &middot;
  <b>Wei Sui</b><sup>2,&dagger;</sup> &middot;
  <b>Xinggang Wang</b><sup>1,&Dagger;</sup>
</p>

<p>
  <sup>1</sup>Huazhong University of Science and Technology &middot;
  <sup>2</sup>D-Robotics &middot;
  <sup>3</sup>Wuhan University &middot;
  <sup>4</sup>Horizon Robotics
</p>

<p>
  <sup>*</sup>Equal contribution &middot;
  <sup>&dagger;</sup>Project Lead &middot;
  <sup>&Dagger;</sup>Corresponding Author
</p>

<a href="https://hustvl.github.io/DreamWAM/"><img src="https://img.shields.io/badge/Project-Page-087f79" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2608.04996"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper arXiv"></a>
<a href="https://huggingface.co/hustvl/DreamWAM"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ffcc4d" alt="Models on Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0 License"></a>

</div>

This repository provides the official implementation of DreamWAM, which moves
world action modeling beyond RGB by learning future appearance, motion,
geometry, and semantics as complementary views of action-relevant state. It
combines joint RGB-flow latent denoising with gated depth and DINO residual
supervision, transferring structured future cues to action prediction through
shared VideoDiT-ActionDiT attention. These beyond-RGB signals are used only
during training, preserving RGB-only inference while improving robustness to
visual perturbations.

## Contents

- [Installation](#installation)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Installation

Run all commands from the repository root. Python 3.10 and CUDA 12.8 are
recommended.

```bash
git clone https://github.com/hustvl/DreamWAM.git
cd DreamWAM

conda create -n dreamwam python=3.10 -y
conda activate dreamwam
conda install -c conda-forge imagemagick -y

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Model Preparation

Place the following upstream repositories and pretrained components at the
repository-local targets shown below.

| Component | Upstream source or weights | Local target |
| --- | --- | --- |
| DreamWAM checkpoints | [Hugging Face](https://huggingface.co/hustvl/DreamWAM) | `checkpoints/` |
| Wan2.2 TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `pretrained/Wan2.2-TI2V-5B/` |
| RAFT | [source](https://github.com/princeton-vl/RAFT) and [weights](https://github.com/princeton-vl/RAFT/blob/master/download_models.sh) | `third_party/RAFT/`, `pretrained/raft-things.pth` |
| DINOv2 | [source](https://github.com/facebookresearch/dinov2) and [ViT-B/14 register weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) | `third_party/dinov2/`, `pretrained/dinov2_vitb14_reg4_pretrain.pth` |
| Depth Anything 3 | [source](https://github.com/ByteDance-Seed/Depth-Anything-3) and [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | `third_party/Depth-Anything-3/`, `pretrained/da3-base/` |
| LIBERO | [benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) | `third_party/LIBERO/` |
| LIBERO-Plus | [benchmark](https://github.com/sylvestf/LIBERO-plus) and [assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus) | `third_party/LIBERO-Plus/`; extract assets to `third_party/LIBERO-Plus/libero/libero/assets/` |

Install the local benchmark packages after placing their source repositories:

```bash
pip install -e third_party/Depth-Anything-3
pip install -e third_party/LIBERO
```

Download the released checkpoints directly into the configured local directory:

```bash
hf download hustvl/DreamWAM \
  dreamwam_joint.pt dreamwam_uncond.pt \
  --local-dir checkpoints
```

Generate the ActionDiT initialization from Wan2.2 before training:

```bash
python scripts/prepare_action_dit.py --config configs/dreamwam_joint.yaml
```

The generated initialization is written to
`pretrained/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.

## Data Preparation

DreamWAM uses the four-suite, LeRobot v2.1 LIBERO release prepared by FastWAM:
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam).
The dataset was prepared with MuJoCo 3.3.2; use the same version for benchmark
consistency.

Download and extract the four suite archives under `data/libero`:

```bash
hf download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --include "*.tar.gz" \
  --local-dir data/libero

for archive in data/libero/*.tar.gz; do
  tar -xzf "$archive" -C data/libero
done
```

The resulting layout must contain:

```text
data/libero/
|-- libero_10_no_noops_lerobot/
|-- libero_goal_no_noops_lerobot/
|-- libero_object_no_noops_lerobot/
`-- libero_spatial_no_noops_lerobot/
```

Precompute the RGB, optical-flow, DINO, and Depth training cache:

```bash
python scripts/precompute_cache.py --config configs/dreamwam_joint.yaml
```

The cache is written to `cache/libero_2cam224` and is shared by both released
training settings.

## Training

```bash
accelerate launch --num_processes 8 scripts/train.py --config configs/dreamwam_uncond.yaml
accelerate launch --num_processes 8 scripts/train.py --config configs/dreamwam_joint.yaml
```

Training writes the resulting checkpoints to `outputs/uncond/checkpoint.pt` and
`outputs/joint/checkpoint.pt`, respectively. Evaluation loads the checkpoint
specified by `paths.checkpoint` in the selected YAML config; point that field
to the corresponding training checkpoint above when evaluating a newly trained
model.

## Evaluation

### LIBERO

Run one of the four official suites by selecting `libero_spatial`,
`libero_object`, `libero_goal`, or `libero_10`:

```bash
python scripts/eval_libero.py \
  --config configs/dreamwam_joint.yaml \
  --suite libero_spatial
```

### LIBERO-Plus

The default command runs the official four-suite protocol with one trial per
task:

```bash
python scripts/eval_libero_plus.py --config configs/dreamwam_joint.yaml
```

Evaluation results are written as JSON under the corresponding
`outputs/<setting>/evaluation/` directory.

## Acknowledgments

DreamWAM is built on the
[FastWAM](https://github.com/yuantianyuan01/FastWAM.git) codebase and its coupled
VideoDiT-ActionDiT formulation. We thank the FastWAM authors for releasing the
base model, training pipeline, and processed LIBERO data.

We also acknowledge the upstream
[Wan2.2](https://github.com/Wan-Video/Wan2.2),
[RAFT](https://github.com/princeton-vl/RAFT),
[DINOv2](https://github.com/facebookresearch/dinov2),
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) projects used by this
release.

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{yuan2026dreamwam,
  title={DreamWAM: Beyond RGB Future Prediction for World Action Models},
  author={Yuan, Shanglin and Zhao, Weiheng and Shi, Xin and Jiang, Haoyi and Guo, Xianda and Liu, Liu and Liu, Wenyu and Sui, Wei and Wang, Xinggang},
  journal={arXiv preprint arXiv:2608.04996},
  year={2026}
}
```
