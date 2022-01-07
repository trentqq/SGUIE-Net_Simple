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