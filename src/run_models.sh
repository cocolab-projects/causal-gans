
screen -S cf_inf -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --cf_inf; exec bash";
screen -S wg+c -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --wass --classifier; exec bash";
screen -S c -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train.py --classifier; exec bash";
