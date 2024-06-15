## A Deep Learning System for Population-level Silent Brain Infarction Detection and Stroke Risk Prediction from Retinal Photographs


Official repo for [A Deep Learning System for Population-level Silent Brain Infarction Detection and Stroke Risk Prediction from Retinal Photographs]

Please contact **njiang2021@sjtu.edu.cn** if you have questions.


### Install environment


This software requires a **Linux** system: [**Ubuntu 22.04**](https://ubuntu.com/download/desktop) or  [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) (other versions are not tested), **16GB memory** and **20GB disk** storage (we recommend 32GB memory). 

1. Create environment with conda:

```
conda create -n deepstroke python=3.7.5 -y
conda activate deepstroke
```

2. Install dependencies

```
git clone https://github.com/njiang2024/DeepSTROKE/
cd DeepSTROKE
pip install -r requirement.txt
```


### Preparing the dataset


JPG and PNG formats are recommended. The input image must be 3-channel color fundus images. For training and testing the fundus model, fundus images should be stored in the input file folder, and the name of each image (e.g. "train_1.jpg") should be filled in the CSV file. Examples of CSV file are shown below:


**CSV file for SBI detection task**
|patient id|gender|ever-smoker at baseline|baseline hypertension|baseline t2dm|baseline age|baseline SBP|baseline HDL-c|baseline total cholesterol|baseline BMI|eye1|eye2|diagnosis|
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| 0  | 1 | 0 | 0 | 0 | 50 | 139 | 1.3 | 4.9 | 24.5 | train_1.jpg | train_2.jpg | 0 |


**CSV file for incident stroke prediction task**
|patient id|gender|ever-smoker at baseline|baseline hypertension|baseline t2dm|baseline age|baseline SBP|baseline HDL-c|baseline total cholesterol|baseline BMI|eye1|eye2|progression_year_1|progression_year_2|progression_year_3|progression_year_4|progression_year_5|
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 0  | 1 | 0 | 0 | 0 | 50 | 139 | 1.3 | 4.9 | 24.5 |train_1.jpg | train_2.jpg | 0 | 0 | 0 | 0 | 1 |

For development of the DeepRETStroke System, 4 csv file should be prepared in the **data** folder. **df_sb_tr.csv** is the training set for SBI Detection, **df_sb_va.csv** is the testing set for SBI Detection, **df_st_tr.csv** is the training set for Stroke Prediction and 
**df_st_va.csv** is the testing set for Stroke Prediction,

### Training for DeepRETStroke System

Run the following command in the terminal to start the system development. **--data_path** should be changed to the custom path of the fundus images. **--finetune** should be set as the primitive encoder of the system, of which the default setting **RETFound_cfp_weights.pth** is an open-source pre-trained model RETFound, a pre-trained foundation model for generalizable disease detection. The weight of RETFound can be downloaded [here](https://github.com/rmaphoh/RETFound_MAE).

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16_s \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path ./data/ \
    --task ./deepstroke/ \
    --finetune ./RETFound_cfp_weights.pth \
    --input_size 256
