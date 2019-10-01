
screen -S cf1 -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train.py --cf_inf --lrn_perturb --seed=100; exec bash";
