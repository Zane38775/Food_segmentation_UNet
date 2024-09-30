# Food Segmentation with U-Net

This project implements a food segmentation model using the U-Net architecture. It's designed to accurately segment food items in images, utilizing the FoodSeg103 dataset.

## Project Structure
![屏幕截图 2024-09-30 101438](https://github.com/user-attachments/assets/12cab844-9205-4089-9e1b-e6c4b212e1be)

## Model Architecture

The project uses a U-Net architecture for food segmentation, implemented in `food_seg.py`.

### U-Net Features:

- Customizable input and output channels (default: 3 input channels, 104 output channels)
- Flexible feature map sizes in the encoding and decoding paths
- Double convolution blocks with batch normalization and ReLU activation
- Skip connections for preserving spatial information
- Upsampling using transposed convolutions

### Model Structure:

1. **Encoder (Downsampling path):**
   - Series of double convolution blocks followed by max pooling
   - Feature map sizes: [64, 128, 256, 512]

2. **Bottleneck:**
   - Double convolution with the highest number of feature maps

3. **Decoder (Upsampling path):**
   - Transposed convolutions for upsampling
   - Concatenation with skip connections
   - Double convolution blocks

4. **Output:**
   - Final 1x1 convolution to map to the number of classes (104 for FoodSeg103)

## Dataset

The `FoodSegDataset` class in `data_prep.py` handles data loading and preprocessing for the FoodSeg103 dataset.

### Dataset Features:
- Supports train/val splits defined in text files
- Loads images and corresponding masks
- Applies customizable data augmentations

### Data Augmentation:
- Training data augmentations include:
  - Resizing
  - Rotation (up to 35 degrees)
  - Horizontal flip (50% chance)
  - Vertical flip (10% chance)
  - Normalization
- Validation data only undergoes resizing and normalization

## Inference

The main segmentation process is implemented in `main_seg.py`.

### Features:
- Loads a trained model from 'trained_model2.pth'
- Processes images from the 'result/input/' folder
- Generates and saves segmentation results in 'result/output/'
- Supports various image formats (PNG, JPG, JPEG)
- Resizes input images to 256x256
- Uses a random color map for visualizing different food classes

## Usage

### Dataset Preparation:
```python
from dataset import FoodSegDataset, get_train_transform, get_val_transform

data_dir = 'path/to/FoodSeg103'
img_dir = 'path/to/FoodSeg103/Images/img_dir'
mask_dir = 'path/to/FoodSeg103/Images/ann_dir'

train_transform = get_train_transform(256, 256)
train_dataset = FoodSegDataset(data_dir, img_dir, mask_dir, split='train', transform=train_transform)

val_transform = get_val_transform(256, 256)
val_dataset = FoodSegDataset(data_dir, img_dir, mask_dir, split='val', transform=val_transform)
```

### Model Initialization:
```python
from unet import UNET
model = UNET(in_channels=3, out_channels=104)
```

### Inference:
1. Place your input images in the result/input/ folder.
2. Run the segmentation script:
```python
python src/main_seg.py
```
3. The segmented images will be saved in the result/output/ folder.

## Requirements

1. Python
2. PyTorch
3. torchvision
4. OpenCV (cv2)
5. Pillow
6. numpy
7. albumentations
   
Install the required packages using:
```python
pip install torch torchvision opencv-python Pillow numpy albumentations
```

## Running the Project

1. Prepare your dataset in the `data/FoodSeg103/` directory.
   Please download the file from url and unzip the data in ./data folder (./data/FoodSeg103/), with passwd: LARCdataset9947
   https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
3. Download a trained model saved as ''trained_model2.pth'' in the project root.
   https://drive.google.com/file/d/1M-QkgFxW-0oHIg_csrhCQ-iOPwtuXpxT/view?usp=sharing
4. Place input images for segmentation in `result/input/`.
5. Run `main_seg.py` to perform segmentation.
6. Check the segmented images in `result/output/`.'

