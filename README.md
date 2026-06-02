# FastTrackTr

<p align="center">
  <strong>FastTrackTr: Fast Multi-Object Tracking with Transformer Trajectory Modeling</strong>
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11277330">📄 IEEE Paper</a> |
  <a href="https://arxiv.org/abs/2411.15811">📑 arXiv</a> |
  <a href="https://drive.google.com/drive/folders/1ROGz1cEKehdm2f2GPW4YeJRTDAcZ8iJD?usp=drive_link">🎬 Demo Videos</a>
</p>

---

## News

- **[2026/06]** Code released with bug fixes.
- **[2026]** Paper published on [IEEE Xplore](https://ieeexplore.ieee.org/document/11277330).

## Overview

FastTrackTr is a fast and efficient multi-object tracking framework built on transformer architectures. It leverages trajectory modeling with RT-DETR as the detection backbone for real-time multi-object tracking on benchmarks such as DanceTrack, SportsMOT, and MOT17.

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8

### Setup

```bash
# Clone the repository
git clone https://github.com/liaopan-lp/fasttracktr.git
cd fasttracktr

# Install dependencies
pip install torch torchvision
pip install einops scipy pyyaml tqdm wandb
```

### Data Preparation

Prepare the datasets (DanceTrack, SportsMOT, MOT17, etc.) and organize them under a `datasets/` directory (or specify the path via `--data-root`):

```
datasets/
├── DanceTrack/
│   ├── train/
│   ├── val/
│   └── test/
├── SportsMOT/
│   ├── train/
│   ├── val/
│   └── test/
└── MOT17/
    ├── train/
    └── test/
```

## Training

### Train on DanceTrack

```bash
python main.py --mode train \
    --config-path ./configs/r50_rt_detr_rtmot_dancetrack.yaml \
    --data-root ./datasets/ \
    --device cuda
```

### Resume Training

```bash
python main.py --mode train \
    --config-path ./configs/r50_rt_detr_rtmot_dancetrack.yaml \
    --resume-model ./output/checkpoint_13.pth \
    --resume-optimizer True \
    --resume-scheduler True \
    --resume-states True
```

## Evaluation

```bash
python main.py --mode eval \
    --config-path ./configs/r50_rt_detr_rtmot_dancetrack.yaml \
    --inference-model ./output/checkpoint_13.pth \
    --inference-dataset DanceTrack \
    --inference-split val
```

## Submission / Inference

```bash
python main.py --mode submit \
    --config-path ./configs/r50_rt_detr_rtmot_dancetrack.yaml \
    --inference-model ./output/checkpoint_13.pth \
    --inference-dataset DanceTrack \
    --inference-split test
```

## Project Structure

```
fasttracktr/
├── main.py                 # Entry point
├── train_engine.py         # Training logic
├── eval_engine.py          # Evaluation logic
├── submit_engine.py        # Submission / inference logic
├── configs/                # YAML configuration files
│   ├── r50_rt_detr_rtmot_dancetrack.yaml
│   ├── r18_fasttracktr_dancetrack.yaml
│   └── ...
├── models/                 # Model implementations
│   ├── fasttracktr.py      # FastTrackTr main model
│   ├── criterion.py        # Loss functions
│   ├── rt_detr_cross/      # RT-DETR with cross-attention
│   └── ...
├── data/                   # Dataset and data loading
├── structures/             # Data structures (Instances, Args, etc.)
├── utils/                  # Utility functions
├── log/                    # Logging utilities
└── TrackEval/              # Evaluation toolkit (HOTA, CLEAR, Identity)
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{fasttracktr2026,
  author={Liao, Pan and Yang, Feng and Wu, Di and Yu, Jinwen and Li, Xingxin and Zhang, Dingwen},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={FastTrackTr: Real-Time Multiobject Tracking With Transformers for Real World}, 
  year={2026},
  volume={22},
  number={3},
  pages={1817-1827},
  keywords={Transformers;Decoding;Real-time systems;Accuracy;Target tracking;Computer architecture;Computational modeling;Feature extraction;Training;Object recognition;Multiobject tracking (MOT);real-time;transformers},
  doi={10.1109/TII.2025.3631698}}
```

## License

This project is released under the [Apache 2.0 License](LICENSE.txt).

## Acknowledgements

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
