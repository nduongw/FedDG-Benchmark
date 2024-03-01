CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pas --test_domains c --method conststyle-bn --num_epoch 200 --dataset pacs --option VGMM-augment

# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method conststyle-bn --num_epoch 50 --style_idx 2 --dataset pacs --option GMM 

# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method conststyle-bn --num_epoch 50 --style_idx 3 --dataset pacs --option GMM 
