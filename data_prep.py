import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FoodSegDataset(Dataset):
    def __init__(self, data_dir, img_dir, mask_dir, split='train', transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        split_file = os.path.join(data_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.img_list = f.read().splitlines()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask, img_name

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

if __name__ == "__main__":
    data_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103'
    img_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\img_dir'
    mask_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\ann_dir'

    train_ds = FoodSegDataset(data_dir, img_dir, mask_dir, split='train', transform=get_train_transform())
    val_ds = FoodSegDataset(data_dir, img_dir, mask_dir, split='val', transform=get_val_transform())

    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")

    image, mask, img_name = train_ds[0]
    print(f"图像形状: {image.shape}")
    print(f"掩码形状: {mask.shape}")
    print(f"图像名称: {img_name}")