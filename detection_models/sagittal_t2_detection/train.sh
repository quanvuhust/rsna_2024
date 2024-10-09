CUDA_VISIBLE_DEVICES=1 python train.py --config configs/custom/fold_0.py --deterministic
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/custom/fold_1.py --deterministic
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/custom/fold_2.py --deterministic
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/custom/fold_3.py --deterministic
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/custom/fold_4.py --deterministic