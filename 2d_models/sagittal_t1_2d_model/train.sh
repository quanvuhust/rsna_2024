export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 code/train.py --exp exp_12
