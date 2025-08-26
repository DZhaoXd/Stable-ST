# Stable-ST
Stable Self-Training for Source-Free Domain Adaptive Semantic Segmentation

This is a pytorch implementation of Stable-ST. 

Stable-ST, a unified framework based on stable sample self-training, incorporates two key technologies: [Dynamic Teacher Update](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Towards_Better_Stability_and_Adaptability_Improve_Online_Self-Training_for_Model_CVPR_2023_paper.html)(CVPR-23 Highlight) and [Stable Neighbor Denoising](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Stable_Neighbor_Denoising_for_Source-free_Domain_Adaptive_Segmentation_CVPR_2024_paper.pdf)(CVPR-24). 

### New perspective
1. We seamlessly integrate the Dynamic Teacher Update mechanism and the Stable Neighbor Denoising technique into a unified framework, Stable-ST. This unified framework allows us to jointly enhance the stability and adaptability of the model in SFDA segmentation tasks.  Moreover, most of the terminology, as well as all figures and experiments, have been reorganized and rewritten to align with the proposed Stable-ST. 

2. We provide a more thorough analysis and experimental validation of the error accumulation issue in self-training for source-free domain adaptation (SFDA) in semantic segmentation tasks,

3. We delve deeper into the DTU technique from stable samples.

4. We provide more detailed experimental support for the neighbor retrieval scheme in the SND technique.

5. We further validate the effectiveness of our method in more realistic SFDA scenarios, including continual SFDA, medical imaging, and remote sensing.



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

### Getting started
Data:
- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

The data folder should be structured as follows:
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/		
...
```
Pretrain models:
- Download pretrained model on GTA5: ([GTA5_NO_DG](https://drive.google.com/file/d/1C_SC1_Ne1r3iqKxjY17wKVHQLRaBq5hT/view?usp=drive_link)) and ([GTA5_DG](https://drive.google.com/file/d/1fZ1uAPxUxPaWQrjBwZ6qkwsY3n2odqYd/view?usp=drive_link)) 
- Download pretrained model on SYNTHIA: ([SYNTHIA_NO_DG](https://drive.google.com/file/d/1380-cAcVxIgyhKWHtf5IGkGbdQGZ7Gzb/view?usp=drive_link)) and ([SYNTHIA_DG](https://drive.google.com/file/d/1_EhjzkcVClC_cjnar6r_tnpU3ZMB8nXG/view?usp=drive_link)) 

Then, put these *.pth into the pretrain folder.

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


### Evaluate
```
# test synthia
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ../DTST/results/synthia_HARD_PL_DTST/model_iter010999.pth > logs/eval_synthia 2>&1 &
# test gta pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_gta_19.yaml resume ./pretrain/G2C_model_iter020000.pth > logs/eval_gta5_pretrain 2>&1 &
# test synthia pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ./pretrain/S2C_Pretrain_NO_DG.pth > logs/eval_synthia_pretrain 2>&1 &
```

- 
