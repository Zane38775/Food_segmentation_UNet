import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FoodSegDataset(Dataset):
    def __init__(self, data_dir, img_dir, mask_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.split = split
        self.transform = transform

        split_file = os.path.join(data_dir, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_list = [line.strip() for line in f]
        
        if not self.image_list:
            raise ValueError(f"No images found in split file: {split_file}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask, img_name

def get_train_transform(image_height, image_width):
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

def get_val_transform(image_height, image_width):
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

# 可以添加一个简单的测试函数来验证数据集的功能
def test_dataset():
    data_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103'
    img_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\img_dir'
    mask_dir = r'C:\Users\zchengam\Documents\UROP24_FoodSeg\data\FoodSeg103\Images\ann_dir'
    
    transform = get_train_transform(256, 256)
    dataset = FoodSegDataset(data_dir, img_dir, mask_dir, split='train', transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    image, mask, img_name = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image name: {img_name}")

if __name__ == "__main__":
    test_dataset()