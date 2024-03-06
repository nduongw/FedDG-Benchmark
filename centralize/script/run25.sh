CUDA_VISIBLE_DEVICES=0 python main.py --train_domains acs --test_domains p --method conststyle-bn --num_epoch 200 --dataset pacs --option VGMM-test_maxvars_4layers --update_interval 20 --wandb 1 --seed 42

# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method conststyle-bn --num_epoch 50 --style_idx 2 --dataset pacs --option GMM 

# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method conststyle-bn --num_epoch 50 --style_idx 3 --dataset pacs --option GMM 
