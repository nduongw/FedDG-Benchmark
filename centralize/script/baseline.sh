#pacs 
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pac --test_domains s --method baseline --num_epoch 200 --dataset pacs --seed 1 --option explore --wandb 0
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pac --test_domains s --method baseline --num_epoch 200 --dataset pacs 
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method baseline --num_epoch 200 --dataset pacs 
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pcs --test_domains a --method baseline --num_epoch 200 --dataset pacs 
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains acs --test_domains p --method baseline --num_epoch 200 --dataset pacs 

#officehome
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains acp --test_domains r --method baseline --num_epoch 200
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains acr --test_domains p --method baseline --num_epoch 200
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains arp --test_domains c --method baseline --num_epoch 200
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains rpc --test_domains a --method baseline --num_epoch 200