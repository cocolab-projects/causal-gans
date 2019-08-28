
screen -S 1 -dm bash -c "CUDA_VISIBLE_DEVICES=0 python train.py --cf_inf --lrn_perturb; exec bash";
screen -S 01 -dm bash -c "CUDA_VISIBLE_DEVICES=1 python train.py --cf_inf --class_loss_wt=.01 --lrn_perturb; exec bash";
screen -S 10000 -dm bash -c "CUDA_VISIBLE_DEVICES=2 python train.py --cf_inf --class_loss_wt=10000 --lrn_perturb; exec bash";
screen -S 100000 -dm bash -c "CUDA_VISIBLE_DEVICES=9 python train.py --cf_inf --class_loss_wt=100000 --lrn_perturb; exec bash";
