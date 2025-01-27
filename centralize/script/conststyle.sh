#pacs 
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pac --test_domains s --method mixstyle --num_epoch 50
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method mixstyle --num_epoch 50
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pcs --test_domains a --method mixstyle --num_epoch 50
# CUDA_VISIBLE_DEVICES=0 python main.py --train_domains acs --test_domains p --method mixstyle --num_epoch 50

#officehome
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains acp --test_domains r --method conststyle-bn --style_idx 1 --num_epoch 50
CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains acp --test_domains r --method conststyle-bn --num_epoch 200
CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains acr --test_domains p --method conststyle-bn --num_epoch 200
CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains arp --test_domains c --method conststyle-bn --num_epoch 200
CUDA_VISIBLE_DEVICES=1 python main.py --dataset officehome --train_domains rpc --test_domains a --method conststyle-bn --num_epoch 200