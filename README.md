# Set up environment
Our model and code are based on CBNetV2 of [UniverseNet](https://github.com/shinya7y/UniverseNet)

```
git clone https://github.com/haiminh2001/Train4Ever.git
```

```
pip install -U openmim
mim install mmcv-full
```

```
cd Train4Ever
pip install -v -e .
```

Change permission to train

```
chmod 777 ./tools/dist_train.sh
```

# Preprocessing and Gen label format COCO

## Preprocessing images to PNG
```
python preprocessing.py --input_path <path_to_input_data> --output_path ./data/TrainImagesPNG
```

## Gen label fomart COCO
```
python genLabelCocoFormat.py --input_labels_path <path_to_input_label_images> --output_folder_path ./data
```

# Our preprocessed data, generated annotation file and pretrain model CBNetV2 on dataset COCO

To save time you can download our preprocessed data and generated annotation file [here](https://drive.google.com/drive/folders/1vXAGZr7MMzyY-gDTfFR0ECH38BeKD1d_?usp=sharing)

Pretrain model CBNetV2 on dataset COCO [here](https://drive.google.com/drive/folders/1MZRwGBnHPyLUeyazakVG-WHaZfcPsGuD?usp=sharing), **put file weight pretrain at folder pretrains**

After the download is complete you set up folder as picture below and put the corresponding dataset

![setup folder](https://github.com/anhnch30820/Train4Ever/blob/master/setupfolder.png)


# Training
```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_780-1100_adamw_3x_coco.py 1
```

You can change file config [here](https://github.com/anhnch30820/Train4Ever/blob/master/configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_780-1100_adamw_3x_coco.py) 

# Inference

checkpoint our model [here](https://drive.google.com/drive/folders/1cFBQSOXAtmpCAKeCePIpeReR884EUbOx?usp=sharing)
```
python Infer.py --input_path <path_to_input_data> --config_path <path_to_input_config> --ckpt_path <path_to_input_ckpt> --output_path <path_to_output>
```

