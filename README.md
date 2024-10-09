# rsna_2024
# dataset
## Axial detection model dataset
https://www.kaggle.com/code/quan0095/rsna-2024-axial-crop-dataset
### Axial detection split dataset
https://www.kaggle.com/code/quan0095/rsna-2024-split-data-axial-detection-model
## 2d and 3d model dataset
https://www.kaggle.com/code/quan0095/rsna-spine-dicom-to-2-5d-jpg/output
### 2d split data
https://www.kaggle.com/code/quan0095/rsna-2024-split-data-2d-model/
### 3d split data
https://www.kaggle.com/code/quan0095/rsna-split-data
# training
## Axial detection model

## 2d model 
### Pretrain
```
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/pre_train.py --exp exp_0_pretrain
```
### 2d model
```
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/train.py --exp exp_7
```
## 3d model
```
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/train.py --exp exp_48
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/finetune_softlabel.py --exp exp_48
```

CV_subarticular_loss:  0.5692208733202356 N_EVAL = 6
CV_subarticular_loss:  0.5664019299946483 N_EVAL = 8
