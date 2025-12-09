<!-- 
This project contains three audio source seperation models.

## 
Document structure

inputs/ audio test input

results/ pretrained models

test_separation_results/   audio separation output 

scripts /contains plot scripts -->

# Audio Source Separation Models

This project includes **three audio source separation models** for music source separation tasks.  
<!-- It supports training**, inference, and evaluation on datasets such as MUSDB18. -->



## Structure

```text
├── input/                    # audio test input
├── results/                  # pretrained models
├── test_separation_results/  # audio separation output 
├── scripts/                  # plot scripts 
```



## Run Command

**To train the model**

```
python3 train.py     --model_type scnet     --config_path configs/config_musdb18_scnet.yaml     --results_path results/     --data_path 'datasets/train' --valid_path datasets/test  \     --num_workers 4     --device_ids 0 --start_check_point results/scnet_checkpoint_musdb18.ckpt
```

**Get separation output**

```
python inference.py \         
    --model_type mdx23c \
    --config_path configs/config_musdb18_mdx23c.yaml \
    --start_check_point results/model_mdx23c.ckpt \
    --input_folder input/ \
    --store_dir separation_results/
```

**Valid model**

```
python3 valid.py --model_type scnet --config_path configs/config_musdb18_scnet.yaml --start_check_point results/scnet_checkpoint_musdb18.ckpt --valid_path datasets/test/ --metrics sdr
```
