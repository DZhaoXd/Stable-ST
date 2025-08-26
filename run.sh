
# G2C
CUDA_VISIBLE_DEVICES=3 nohup python train_stable_st.py -cfg configs/deeplabv2_r101_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST/ resume pretrain/G2C_model_iter020000.pth > logs/G2C_StableST.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train_stable_st.py -cfg configs/segformer_mitb5_StableST_G2C.yaml OUTPUT_DIR results/G2C_StableST_Segf_mitb5/ resume pretrain/G2C_model_iter020000_Segf_mitb5.pth > logs/G2C_StableST_Segf_mitb5.file 2>&1 &

## G2S
CUDA_VISIBLE_DEVICES=0 nohup python train_stable_st.py -cfg configs/deeplabv2_r101_StableST_S2C.yaml OUTPUT_DIR results/S2C_StableST/ resume pretrain/S2C_model_iter020000.pth > logs/S2C_StableST 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python train_stable_st.py -cfg configs/segformer_mitb5_StableST_S2C.yaml OUTPUT_DIR results/S2C_StableST_Segf_mitb5/ resume pretrain/G2C_model_iter020000_Segf_mitb5.pth > logs/S2C_StableST_Segf_mitb5.file 2>&1 &

## BDD
CUDA_VISIBLE_DEVICES=6 nohup  python train_stable_st.py -cfg configs/deeplabv2_r101_dtst_BDD.yaml OUTPUT_DIR results/BDD_SND_WARM/ resume pretrain/G2C_model_iter020000.pth > logs/BDD_SND_WARM.file 2>&1 &

# test synthia
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ../DTST/results/synthia_HARD_PL_DTST/model_iter010999.pth > logs/eval_synthia 2>&1 &
# test gta pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_gta_19.yaml resume ./pretrain/G2C_model_iter020000.pth > logs/eval_gta5_pretrain 2>&1 &
# test synthia pretrain
CUDA_VISIBLE_DEVICES=3 nohup python test.py -cfg configs/eval_synthia_16.yaml resume ./pretrain/S2C_Pretrain_NO_DG.pth > logs/eval_synthia_pretrain 2>&1 &

