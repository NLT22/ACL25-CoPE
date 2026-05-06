# ACL25-CoPE
The official code for [ACL 2025 Oral: Modeling Uncertainty in Composed Image Retrieval via Probabilistic Embeddings](https://aclanthology.org/2025.acl-long.61/).

# Outline
We apply a probabilistic embedding approach to mitigate the data uncertainty issues within composed image retrieval.
We design a novel probabilistic learning approach including a and a hierarchical learning objective that:

- Penalizes high uncertainty values during matching. This improves training robustness and also prioritizes more confident matching during inference.
- Captures the mutual cancelation effect within CIR instructions: ignore target uncertainties in aspects where query is also uncertain.

We hope our approach can provide new insights on how to enable multi-modal retrieval models to identify and mitigate uncertainty without sophisticated model architecture and extra data.

# Training Guidelines

## Environment Setup

```
cd ACL25-CoPE/ && pip install -e .
```

## Data Preparation

Follow the instructions of [fashion-iq](https://github.com/XiaoxiaoGuo/fashion-iq) and [CIRR](https://github.com/Cuberick-Orion/CIRR) datasets, and organize dataset files in the following format.

### Fashion-IQ

```
fashion-iq/
├── images/
│   ├── [image_name].png
│   └── ...
├── captions/
│   ├── cap.dress.train.json
│   ├── cap.dress.val.json
│   ├── cap.dress.test.json
│   ├── cap.shirt.train.json
│   ├── cap.shirt.val.json
│   ├── cap.shirt.test.json
│   ├── cap.toptee.train.json
│   ├── cap.toptee.val.json
│   └── cap.toptee.test.json
└── image_splits/
    ├── split.dress.train.json
    ├── split.dress.val.json
    ├── split.dress.test.json
    ├── split.shirt.train.json
    ├── split.shirt.val.json
    ├── split.shirt.test.json
    ├── split.toptee.train.json
    ├── split.toptee.val.json
    └── split.toptee.test.json
```

### CIRR

```
CIRR/
├── captions/
│   ├── cap.rc2.train.json
│   ├── cap.rc2.val.json
│   └── cap.rc2.test1.json
├── image_splits/
│   ├── split.rc2.train.json
│   ├── split.rc2.val.json
│   └── split.rc2.test1.json
├── test1/
├── dev/
└── train/
    ├── [numbered_directories_0-99]/
    └── ...
```

## Configuration

Finish model, training, and inference configuration. Refer to the example in `config_example.yaml` for recommended settings. Typically one would want to specify the ouput path and the target dataset, and adjust the batch size according to their computing resources.

## Run training on a single GPU

Simply run 

```
python train.py -c /path/to/your/config.yaml
```

## Run training on multiple GPUs

Use PyTorch DistributedDataParallel through `torchrun`. The batch size in the
config is per GPU, so the effective batch size is `batch_size * nproc_per_node`.

```
torchrun --standalone --nproc_per_node=2 train.py -c /path/to/your/config.yaml
```

For Kaggle's 2-GPU notebooks, keep `device: cuda:0` in the config and launch
with `torchrun`; each process will automatically bind to its own local GPU.
When `training.warmup_epochs > 0`, keep
`distributed_config.find_unused_parameters: true` because the warmup loss does
not use every probabilistic loss module.

# Cite
```
@inproceedings{tang-etal-2025-modeling,
  title     = {Modeling Uncertainty in Composed Image Retrieval via Probabilistic Embeddings},
  author    = {Tang, Haomiao and Wang, Jinpeng and Peng, Yuang and Meng, GuangHao and Luo, Ruisheng and Chen, Bin and Chen, Long and Wang, Yaowei and Xia, Shu-Tao},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = jul,
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.acl-long.61/},
  doi       = {10.18653/v1/2025.acl-long.61},
  pages     = {1210--1222},
  isbn      = {979-8-89176-251-0}
}
```
