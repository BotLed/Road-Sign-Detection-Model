import cv2
import os
import numpy as np
from pathlib import Path
import yaml

def apply_digital_tape(image, bbox, color=(50, 50, 50), thickness_ratio=0.2):
    """
    Applies a horizontal tape strip across center of the bounding box.
    """
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox

    # converts normalized YOLO coordinates to pixel coordinates
    x1 = int((x_center - b_w / 2) * w)
    y1 = int((y_center - b_h / 2) * h)
    x2 = int((x_center + b_w / 2) * w)
    y2 = int((y_center + b_h / 2) * h)

    # define tape dimensions (horizontal strip across middle)
    tape_h = int((y2 - y1) * thickness_ratio)
    tape_y1 = y_center * h - (tape_h / 2)
    tape_y2 = y_center * h + (tape_h / 2)

    # draw tape
    cv2.rectangle(image, (x1, int(tape_y1)), (x2, int(tape_y2)), color, -1)
    return image

def apply_adversarial_patch(image, bbox):
  h, w, _ = image.shape
  x_center, y_center, b_w, b_h = bbox

  # convert to pixel coordinates
  x1 = int((x_center - b_w / 4) * w)
  y1 = int((y_center - b_h / 4) * h)
  x2 = int((x_center + b_w / 4) * w)
  y2 = int((y_center + b_h / 4) * h)

  # gen adversarial noise
  noise_patch = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)

  # overlay the noise onto sign
  image[y1:y2, x1:x2] = noise_patch
  return image

def apply_heavy_occlusion(image, bbox):
  h, w, _ = image.shape
  x_center, y_center, b_w, b_h = bbox

  # increases tape patch size to 80 percent of bounding box
  x1 = int((x_center - b_w * 0.4) * w)
  y1 = int((y_center - b_h * 0.4) * h)
  x2 = int((x_center + b_w * 0.4) * w)
  y2 = int((y_center + b_h * 0.4) * h)

  cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), -1)
  return image


import shutil
import os
import cv2
import numpy as np


# NOTE: IF YOU ADD A NEW ATTACK TYPE, ADD IT AS A PARAM TO THIS FUNCTION
def generate_attack_dataset(base_path, attack_type="patch"):
    """
    attack_type options: 'patch', 'tape', 'heavy_occlusion'
    """
    # Map string names to the actual functions
    attack_map = {
        "patch": apply_adversarial_patch,
        "tape": apply_digital_tape,
        "heavy_occlusion": apply_heavy_occlusion
    }

    if attack_type not in attack_map:
        print(f"❌ Invalid attack type: {attack_type}. Defaulting to 'patch'.")
        attack_func = apply_adversarial_patch
    else:
        attack_func = attack_map[attack_type]

    val_images_path = os.path.join(base_path, 'valid', 'images')
    val_labels_path = os.path.join(base_path, 'valid', 'labels')
    output_base = os.path.join(base_path, 'attacked_val')
    output_images_path = os.path.join(output_base, 'images')
    output_labels_path = os.path.join(output_base, 'labels')
    
    os.makedirs(output_images_path, exist_ok=True)
    
    if os.path.exists(val_labels_path):
        if os.path.exists(output_labels_path):
            shutil.rmtree(output_labels_path)
        shutil.copytree(val_labels_path, output_labels_path)
        print(f"✅ Labels synced for {attack_type} attack.")

    print(f"Creating '{attack_type}' attacked images...")
    for img_name in os.listdir(val_images_path):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')): continue

        img = cv2.imread(os.path.join(val_images_path, img_name))
        label_file = os.path.join(val_labels_path, img_name.rsplit('.', 1)[0] + '.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        bbox = [float(x) for x in parts[1:]]
                        # Dynamically call the chosen function
                        img = attack_func(img, bbox)
        
        cv2.imwrite(os.path.join(output_images_path, img_name), img)

def create_attack_yaml(base_path):
    """Generates attack.yaml based on the original data.yaml."""
    original_yaml = os.path.join(base_path, 'data.yaml')
    attack_yaml = os.path.join(base_path, 'attack.yaml')

    if os.path.exists(original_yaml):
        with open(original_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        attack_config = {
            'path': 'datasets/archive', 
            'train': 'train/images',
            'val': 'attacked_val/images',
            'test': 'valid/images',
            'nc': data['nc'],
            'names': data['names']
        }

        with open(attack_yaml, 'w') as f:
            yaml.dump(attack_config, f)
        print(f"✅ Generated {attack_yaml}")