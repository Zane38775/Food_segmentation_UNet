import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from food_seg import UNET
import os

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 104

# Load the trained model
model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('trained_model2.pth', map_location=DEVICE))
model.eval()

# Define the image transformation
transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

# Set the input and output folder paths
input_folder = r'C:\Users\Zane\Documents\UROP24_FoodSeg_1\result\input'
output_folder = r'C:\Users\Zane\Documents\UROP24_FoodSeg_1\result\output'

# Get the list of image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image file
for image_file in image_files:
    # Load and preprocess the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transform(image=image)['image']
    input_image = transformed_image.unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        prediction = torch.argmax(output, dim=1)
        prediction = prediction.squeeze().cpu().numpy()

    # Convert the prediction to RGB color map
    color_map = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
    rgb_prediction = color_map[prediction]

    # Save the segmentation result
    output_path = os.path.join(output_folder, f'segmented_{image_file}')
    cv2.imwrite(output_path, cv2.cvtColor(rgb_prediction, cv2.COLOR_RGB2BGR))
    print(f"Segmentation result for '{image_file}' saved as '{output_path}'")