
screen -S cf_inf -dm bash -c "CUDA_VISIBLE_DEVICES=6 python train.py --cf_inf; exec bash";
screen -S wg+c -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train.py --wass --classifier; exec bash";
screen -S wg+ali+c -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train.py --wass --ali --classifier; exec bash";
screen -S c -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train.py --classifier; exec bash";
