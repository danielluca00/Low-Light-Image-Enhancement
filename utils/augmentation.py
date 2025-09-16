import os
import random
import argparse
import cv2
import albumentations as A
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("highFolder", help="Path to high-light images folder.")
parser.add_argument("lowFolder", help="Path to low-light images folder.")
parser.add_argument("augHighFolder", help="Path to save augmented high-light images.")
parser.add_argument("augLowFolder", help="Path to save augmented low-light images.")
parser.add_argument("augPercent", help="Percentage [0-1] of images to augment.")

args = parser.parse_args()

high_folder = args.highFolder
low_folder = args.lowFolder
aug_high_path = args.augHighFolder
aug_low_path = args.augLowFolder
aug_percent = float(args.augPercent)

# ----------------------
# CREATE OUTPUT FOLDERS
# ----------------------
os.makedirs(aug_high_path, exist_ok=True)
os.makedirs(aug_low_path, exist_ok=True)

# ----------------------
# AUGMENTATION SETTINGS
# ----------------------
augmentation_settings = [
    A.VerticalFlip(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    A.MotionBlur(blur_limit=3, p=1.0),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.3), p=1.0)
]

aug_settings_names = [
    'vertical_flip',
    'horizontal_flip',
    'brightness_contrast',
    'gauss_noise',
    'motion_blur',
    'hue_saturation',
    'gamma',
    'clahe',
    'sharpen'
]

# ----------------------
# LIST FILES AND SELECT RANDOM SUBSET
# ----------------------
high_files = sorted([f for f in os.listdir(high_folder) if not f.startswith('.')])
low_files = sorted([f for f in os.listdir(low_folder) if not f.startswith('.')])

assert high_files == low_files, "High-light and low-light file names do not match!"

num_to_augment = int(len(high_files) * aug_percent)
selected_files = random.sample(high_files, num_to_augment)

# ----------------------
# APPLY AUGMENTATIONS
# ----------------------
for file in tqdm(selected_files, desc="Augmenting images"):
    img_high = cv2.imread(os.path.join(high_folder, file))
    img_low = cv2.imread(os.path.join(low_folder, file))

    if img_high is None or img_low is None:
        print(f"Error loading: {file}")
        continue

    # Apply all selected transformations
    for setting, name in zip(augmentation_settings, aug_settings_names):
        augmented_high = setting(image=img_high)['image']
        augmented_low = setting(image=img_low)['image']

        cv2.imwrite(os.path.join(aug_high_path, f'aug_{name}_{file}'), augmented_high)
        cv2.imwrite(os.path.join(aug_low_path, f'aug_{name}_{file}'), augmented_low)

print("Data augmentation completed!")
