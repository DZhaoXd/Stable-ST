# Stable-ST
Stable Self-Training for Source-Free Domain Adaptive Semantic Segmentation

This is a pytorch implementation of Stable-ST. 

Stable-ST, a unified framework based on stable sample self-training, incorporates two key technologies: [Dynamic Teacher Update](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Towards_Better_Stability_and_Adaptability_Improve_Online_Self-Training_for_Model_CVPR_2023_paper.pdf)(CVPR-23 Highlight) and [Stable Neighbor Denoising](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Stable_Neighbor_Denoising_for_Source-free_Domain_Adaptive_Segmentation_CVPR_2024_paper.pdf)(CVPR-24). 

### Prerequisites
- Python 3.6
- Pytorch 1.2.0
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0

### Step-by-step installation

```bash
conda create --name Stable-ST -y python=3.6
conda activate Stable-ST

# this installs the right pip and dependencies for the fresh python
conda install -y python pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

## Getting Started

### Data Preparation

Please download and organize the datasets as follows.
All datasets should be placed under the `data/` directory.

---

#### **Cityscapes**

Please download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from
[https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)

Extract them to:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

---

#### **GTA5**

Please download all GTA5 image and label packages from
[https://download.visinf.tu-darmstadt.de/data/from_games/](https://download.visinf.tu-darmstadt.de/data/from_games/)

Extract them to:

```text
data/GTA5/GTAV/
├── images/
└── labels/
```

---

#### **ACDC**

Please download `rgb_anon_trainvaltest.zip` and `gt_trainval.zip` from
[https://acdc.vision.ee.ethz.ch/download](https://acdc.vision.ee.ethz.ch/download)

Extract them to `data/ACDC/`.
The original directory structure follows `condition/split/sequence/`.
Please reorganize it into a flat `split/` layout as below:

```text
data/ACDC/
├── rgb_anon/
│   ├── train/
│   └── val/
└── gt/
    ├── train/
    └── val/
```

---

#### **BDD100K**

Please download **10K Images** and **Segmentation** from
[https://bdd-data.berkeley.edu/portal.html#download](https://bdd-data.berkeley.edu/portal.html#download)

Extract them to:

```text
data/BDD/bdd100k/
├── images/10k/
│   ├── train/
│   └── val/
└── labels/sem_seg/masks/
    ├── train/
    └── val/
```

---

#### **Mapillary Vistas**

Please download `mapillary-vistas-dataset_public_v1.2.zip` from
[https://www.mapillary.com/dataset/vistas](https://www.mapillary.com/dataset/vistas)

Extract it to:

```text
data/mapillary/
```

---

#### **EndoScene**

Please download the EndoScene dataset from
[https://service.tib.eu/ldmservice/dataset/endoscene](https://service.tib.eu/ldmservice/dataset/endoscene)

Organize it as:

```text
data/EndoScene/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

#### **ETIS-Larib**

Please download the ETIS-Larib Polyp Dataset from
[https://service.tib.eu/ldmservice/dataset/etis-larib-polyp-db](https://service.tib.eu/ldmservice/dataset/etis-larib-polyp-db)

Organize it as:

```text
data/ETIS-Larib/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

#### **Potsdam & Vaihingen**

Please download the ISPRS 2D Semantic Labeling datasets:

* Potsdam: [https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
* Vaihingen: [https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)

Organize them as:

```text
data/potsdam/
├── train_images/
├── train_gt/
├── val_images/
└── val_gt/

data/vaihingen/
├── train_images/
├── train_gt/
├── val_images/
└── val_gt/
```

### 19-Class Label Conversion

For experiments based on the Cityscapes 19-class setting, please convert the datasets using:

```bash
python tools/convert_datasets_to19/gta.py data/GTA5/GTAV
python tools/convert_datasets_to19/cityscapes.py data/cityscapes
python tools/convert_datasets_to19/mapillary.py data/mapillary
```

After conversion, the expected structure is:

```text
data/
├── GTA5/GTAV/
│   ├── images/
│   └── labels_19/
├── cityscapes/
│   ├── leftImg8bit/
│   └── gtFine_19/
├── BDD/
├── ACDC/
├── EndoScene/
├── ETIS-Larib/
├── potsdam/
└── vaihingen/
```

### Train
G2C model adaptation
```
python train_stable_st.py -cfg configs/deeplabv2_r101_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST/ resume pretrain/G2C_model_iter020000.pth 
```
S2C model adaptation

```
python train_stable_st.py -cfg configs/segformer_mitb5_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST_Segf_mitb5/ resume pretrain/G2C_model_iter020000_Segf_mitb5.pth
```
G2B model adaptation

```
python train_stable_st.py -cfg configs/deeplabv2_r101_dtst_BDD.yaml OUTPUT_DIR results/BDD_StableST/ resume pretrain/G2C_model_iter020000.pth 
```


Besides, we still support the Segformer-B5 in StableSt.
For G2C using segformer_mitb5:
```
CUDA_VISIBLE_DEVICES=3 nohup python train_stable_st.py -cfg configs/segformer_mitb5_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST_Segf_mitb5/ resume pretrain/G2C_model_iter020000_Segf_mitb5.pth > logs/G2C_StableST_Segf_mitb5.file 2>&1 &
```

For S2C using segformer_mitb5:
```
python train_stable_st.py -cfg configs/deeplabv2_r101_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST/ resume pretrain/G2C_model_iter020000.pth
```

For G2C using dinoV2_L:
```
python train_lora.py -cfg configs/DinoV2_L_adaptor.yaml OUTPUT_DIR results/DinoV2_L_adaptor
python train_stable_st.py -cfg configs/DinoV2_L_G2C.yaml OUTPUT_DIR results/G2C_StableST_DinoV2_L_adaptor/ resume results/DinoV2_L_adaptor/G_model_iter020000.pth
```

For E2E in Medical images:
```
python train.py -cfg configs/DinoV2_L_adaptor.yaml OUTPUT_DIR results/DinoV2_L_adaptor
python train_stable_st.py -cfg configs/DinoV2_L_G2C.yaml OUTPUT_DIR results/G2C_StableST_DinoV2_L_adaptor/ resume results/DinoV2_L_adaptor/G_model_iter020000.pth
```


### Evaluate
```
# test synthia
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ../DTST/results/synthia_HARD_PL_DTST/model_iter010999.pth > logs/eval_synthia 2>&1 &
# test gta pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_gta_19.yaml resume ./pretrain/G2C_model_iter020000.pth > logs/eval_gta5_pretrain 2>&1 &
# test synthia pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ./pretrain/S2C_Pretrain_NO_DG.pth > logs/eval_synthia_pretrain 2>&1 &
```
