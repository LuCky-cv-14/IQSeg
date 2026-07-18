# IQSeg

IQSeg is a two-stage semantic segmentation framework for infrared-visible image pairs. This repository currently releases the first-stage training and evaluation code, which is built on Detectron2 / MaskFormer. The second-stage code will be released later.

> Note: Some paths in the source code are still local placeholders. Before training or evaluation, please update the dataset, checkpoint, and output paths for your own environment.

## News

- The first-stage training and testing code is available in `IQSeg_s1`.
- The second-stage code in `IQSeg_s2` will be released later.

## Repository Structure

```text
IQSeg/
├── IQSeg_s1/                 # First-stage training and evaluation code
│   ├── configs/              # Config files for FMB, MSRS, and PST900
│   ├── mask_former/          # MaskFormer model, datasets, and utilities
│   ├── train_first_stage.py  # Training/evaluation entry point
│   └── dcnv4.py              # DCNv4 replacement file
├── detectron2/               # Detectron2 code used by this project
└── readme.md
```

## Environment

Recommended environment:

- Python 3.10.13
- CUDA 11.3
- PyTorch 1.11.0
- torchvision 0.12.0
- torchaudio 0.11.0
- Detectron2 0.6

Create and activate the conda environment:

```bash
conda create --name IQSeg python=3.10.13 -y
conda activate IQSeg
```

Install PyTorch:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the local Detectron2 package:

```bash
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.3

cd detectron2
pip install -e .
cd ..
```

If Detectron2 installation fails because of `setuptools`, run:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "setuptools<70"
cd detectron2
pip install -e .
cd ..
```

Install other dependencies:

```bash
pip install timm==0.9.12
pip install scipy==1.13.1
pip install scikit-image==0.22.0
pip install scikit-learn==1.3.2
```

Install Segment Anything only if you need to run the second-stage code:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## DCNv4 Installation

IQSeg uses DCNv4. Please first install DCNv4 by following the official repository:

https://github.com/OpenGVLab/DCNv4

After installation, replace the installed `dcnv4.py` with the file provided in this repository:

```bash
cp IQSeg_s1/dcnv4.py \
  ~/miniconda3/envs/IQSeg/lib/python3.10/site-packages/DCNv4-1.0.0.post2-py3.10-linux-x86_64.egg/DCNv4/modules/dcnv4.py
```

If your conda environment name or DCNv4 installation path is different, update the target path accordingly.

## Dataset

The first-stage code supports infrared-visible semantic segmentation datasets in the FMB, MSRS, and PST900 formats.

Recommended dataset structure:

```text
<dataset_root>/
├── train/
│   ├── IRwarp/
│   ├── Visible/
│   └── Label/
└── test/
    ├── IRwarp/
    ├── Visible/
    └── Label/
```

For each sample, the files under `IRwarp`, `Visible`, and `Label` should share the same filename.

Dataset download:

```text
链接: https://pan.baidu.com/s/1icBXNCqYxJ1M_Vi1dxOYig 
Password: hiyb 
```

## Pretrained Weights and Checkpoints

The first-stage checkpoint can be downloaded from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1LOXFgmzq6ky8x8KYr5KfNw?pwd=hbdh
Password: hbdh
```

Recommended checkpoint structure:

```text
IQSeg/
└── checkpoint/
    └── FMB_first_stage_r50.pth
```

## Paths and Configuration

Model config files are located in:

```text
IQSeg_s1/configs/
├── fmb-stuff-15/
├── msrs-stuff-9/
└── pst-stuff-5/
```

Before training or evaluation, configure the dataset paths, output path, and checkpoint path in:

```text
IQSeg_s1/train_first_stage.py
```

Training path configuration:

```python
def add_path_config_train(cfg):
    cfg.OUTPUT_DIR = "xxx/IQSeg/IQSeg_s1/checkpoint/FMB_r50"
    cfg.TRAIN_PATH = "xxx/RGBTdataset/FMB_Dataset/train"
    cfg.TEST_PATH = "xxx/RGBTdataset/FMB_Dataset/test"
```

Evaluation path configuration:

```python
def add_path_config_test(cfg, i):
    cfg.OUTPUT_DIR = "xxx/IQSeg/IQSeg_s1/output/eval_FMB_first_stage"
    cfg.MODEL.WEIGHTS = "xxx/IQSeg/checkpoint/FMB_first_stage_r50.pth"
    cfg.TRAIN_PATH = "xxx/RGBTdataset/FMB_Dataset/train"
    cfg.TEST_PATH = "xxx/RGBTdataset/FMB_Dataset/test"
```

GPU-related arguments can be configured in:

```text
detectron2/detectron2/engine/defaults.py
```

Common arguments:

```text
--config-file    model config file
--num-gpus       number of GPUs
```

You can also set the visible GPU in `IQSeg_s1/train_first_stage.py`:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

## Dataset Switch

The default dataset class is imported in `IQSeg_s1/train_first_stage.py`.

For FMB / MSRS:

```python
from mask_former.data.datasets.selfdataset import MSRSDataset
```

For PST900:

```python
from mask_former.data.datasets.selfdataset_pst import MSRSDataset
```

## Training

Set the entry function in `IQSeg_s1/train_first_stage.py` to `train`:

```python
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
```

Start training on FMB:

```bash
cd IQSeg_s1
python train_first_stage.py \
  --config-file configs/fmb-stuff-15/maskformer_R50_bs32_60k.yaml \
  --num-gpus 1
```

For MSRS or PST900, switch the config file:

```bash
# MSRS
python train_first_stage.py --config-file configs/msrs-stuff-9/maskformer_R50_bs32_60k.yaml --num-gpus 1

# PST900
python train_first_stage.py --config-file configs/pst-stuff-5/maskformer_R50_bs32_60k.yaml --num-gpus 1
```

## Evaluation

Set the entry function in `IQSeg_s1/train_first_stage.py` to `test_all_models`:

```python
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        test_all_models,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
```

Start evaluation:

```bash
cd IQSeg_s1
python train_first_stage.py \
  --config-file configs/fmb-stuff-15/maskformer_R50_bs32_60k.yaml \
  --num-gpus 1
```

Evaluation results will be saved to `cfg.OUTPUT_DIR`.

## TODO

- Release the second-stage code.
- Add dataset download links.

## Acknowledgement

This project is built on the following excellent open-source projects:

- Detectron2
- MaskFormer
- Segment Anything
- DCNv4

## Citation

If this project is helpful for your research, please cite our paper:

```bibtex
@article{liu2026implicit,
  title={Implicit alignment and query refinement for RGB-T semantic segmentation},
  author={Liu, Chang and Liu, Haizhuang and Zhuo, Junbao and Zou, Bochao and Chen, Jiansheng and Zhao, Qianchuan and Ma, Huimin},
  journal={Pattern Recognition},
  volume={169},
  pages={111951},
  year={2026},
  publisher={Elsevier}
}
```
