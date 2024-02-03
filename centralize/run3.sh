CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pac --test_domains s --method mixstyle --num_epoch 50 --option bn
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pas --test_domains c --method mixstyle --num_epoch 50 --option bn
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains pcs --test_domains a --method mixstyle --num_epoch 50 --option bn
CUDA_VISIBLE_DEVICES=0 python main.py --train_domains acs --test_domains p --method mixstyle --num_epoch 50 --option bn