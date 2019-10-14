
screen -S no_g_loss1 -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train.py --cf_inf --lrn_perturb --seed=10; exec bash";
screen -S no_g_loss2 -dm bash -c "CUDA_VISIBLE_DEVICES=8 python train.py --cf_inf --lrn_perturb --seed=100; exec bash";
screen -S no_g_loss3 -dm bash -c "CUDA_VISIBLE_DEVICES=7 python train.py --cf_inf --lrn_perturb --seed=1000; exec bash";
