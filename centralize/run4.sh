CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pac --test_domains s --method dsu --num_epoch 50
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method dsu --num_epoch 50
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pcs --test_domains a --method dsu --num_epoch 50
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains acs --test_domains p --method dsu --num_epoch 50