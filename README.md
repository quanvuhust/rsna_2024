# rsna_2024
# dataset
https://www.kaggle.com/code/quan0095/rsna-spine-dicom-to-2-5d-jpg/output?scriptVersionId=185361903&select=train.zip
# training
```
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/train.py --exp exp_23
```