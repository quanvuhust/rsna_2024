while ps -p 691545 > /dev/null; do sleep 1; done
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/train.py --exp exp_67
