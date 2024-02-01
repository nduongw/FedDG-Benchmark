CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pac --test_domain s --method mixstyle --num_epoch 50
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pas --test_domain c --method mixstyle --num_epoch 50
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pcs --test_domain a --method mixstyle --num_epoch 50
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains acs --test_domain p --method mixstyle --num_epoch 50


CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pac --test_domain s --method conststyle --num_epoch 50 --style_idx 1
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pac --test_domain s --method conststyle --num_epoch 50 --style_idx 2
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pac --test_domain s --method conststyle --num_epoch 50 --style_idx 3

CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pas --test_domain c --method conststyle --num_epoch 50 --style_idx 1
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pas --test_domain c --method conststyle --num_epoch 50 --style_idx 2
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pas --test_domain c --method conststyle --num_epoch 50 --style_idx 3


CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pcs --test_domain a --method conststyle --num_epoch 50 --style_idx 1
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pcs --test_domain a --method conststyle --num_epoch 50 --style_idx 2
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains pcs --test_domain a --method conststyle --num_epoch 50 --style_idx 3

CUDA_VISIBLE_DEVICES=1 python main.py --train_domains acs --test_domain p --method conststyle --num_epoch 50 --style_idx 1
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains acs --test_domain p --method conststyle --num_epoch 50 --style_idx 2
CUDA_VISIBLE_DEVICES=1 python main.py --train_domains acs --test_domain p --method conststyle --num_epoch 50 --style_idx 3
