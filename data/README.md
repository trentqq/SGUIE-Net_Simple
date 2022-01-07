## Data Preparation
### 1. Download the train + test set of [SUIM-E](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM?usp=sharing).

### 2. The structure of data folder is as follows:
```
├── dataset_name
    ├── train
        ├── raw
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── ref
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── mask
            ├── BW
                ├── im1.bmp
                ├── im2.bmp
                └── ...
            ├── FV
                ├── im1.bmp
                ├── im2.bmp
                └── ...
            ├── ..
    ├── test
        ├── raw
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── mask
            ├── BW
                ├── im1.bmp
                ├── im2.bmp
                └── ...
            ├── FV
                ├── im1.bmp
                ├── im2.bmp
                └── ...
            ├── ..

```

## Custom Datasets(without segmentation map)
For other datasets without ground truth segmentation map, users can choose to:
1. Manually mark the segmentation map
2. Using [SUIM-Net](https://github.com/xahidbuffon/SUIM) to generate predicted semantic segmentation, by loading [pre-trained models](https://drive.google.com/drive/folders/1aoluekvB_CzoaqGhLutwtJptIOBasl7i).

Then, organize the obtained segmentation map as described [above](#Data-Preparation).
