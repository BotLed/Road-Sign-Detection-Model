import cv2
import os
import numpy as np
from pathlib import Path
import yaml
import shutil


# ── Attack functions ─────────────────────────────────────────────────────────

def apply_digital_tape(image, bbox, color=(50, 50, 50), thickness_ratio=0.2):
    """
    Applies a horizontal tape strip across the center of the bounding box.
    """
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox

    x1 = int((x_center - b_w / 2) * w)
    y1 = int((y_center - b_h / 2) * h)
    x2 = int((x_center + b_w / 2) * w)
    y2 = int((y_center + b_h / 2) * h)

    tape_h = int((y2 - y1) * thickness_ratio)
    tape_y1 = y_center * h - (tape_h / 2)
    tape_y2 = y_center * h + (tape_h / 2)

    cv2.rectangle(image, (x1, int(tape_y1)), (x2, int(tape_y2)), color, -1)
    return image


def apply_adversarial_patch(image, bbox):
    """
    Overlays a random noise patch at the center of the bounding box (50% of bbox area).
    """
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox

    x1 = int((x_center - b_w / 4) * w)
    y1 = int((y_center - b_h / 4) * h)
    x2 = int((x_center + b_w / 4) * w)
    y2 = int((y_center + b_h / 4) * h)

    patch_h = max(1, y2 - y1)
    patch_w = max(1, x2 - x1)
    noise_patch = np.random.randint(0, 255, (patch_h, patch_w, 3), dtype=np.uint8)
    image[y1:y2, x1:x2] = noise_patch
    return image


def apply_heavy_occlusion(image, bbox):
    """
    Covers 80% of the bounding box with a solid grey rectangle.
    """
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox

    x1 = int((x_center - b_w * 0.4) * w)
    y1 = int((y_center - b_h * 0.4) * h)
    x2 = int((x_center + b_w * 0.4) * w)
    y2 = int((y_center + b_h * 0.4) * h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), -1)
    return image


def apply_graffiti(image, bbox, num_elements=6):
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox
    x1 = int((x_center - b_w / 2) * w)
    y1 = int((y_center - b_h / 2) * h)
    x2 = int((x_center + b_w / 2) * w)
    y2 = int((y_center + b_h / 2) * h)
    sign_w = max(1, x2 - x1)
    sign_h = max(1, y2 - y1)

    # Skip graffiti if the bounding box is too small to draw on meaningfully
    if sign_w < 12 or sign_h < 12:
        return image

    def rand_point():
        return (int(np.random.uniform(x1, x2)), int(np.random.uniform(y1, y2)))

    def rand_color():
        hue = np.random.randint(0, 180)
        bgr = cv2.cvtColor(np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    overlay = image.copy()
    for _ in range(num_elements):
        color      = rand_color()
        shape_type = np.random.choice(['line', 'filled_rect', 'circle', 'polyline'])
        thickness  = np.random.randint(2, max(3, sign_w // 6))
        if shape_type == 'line':
            cv2.line(overlay, rand_point(), rand_point(), color, thickness)
        elif shape_type == 'filled_rect':
            pt   = rand_point()
            # Clamp range so low < high even on small signs
            rw   = max(2, sign_w // 6)
            rh   = max(2, sign_h // 6)
            rx2  = pt[0] + np.random.randint(rw, max(rw + 1, sign_w // 2))
            ry2  = pt[1] + np.random.randint(rh, max(rh + 1, sign_h // 2))
            cv2.rectangle(overlay, pt, (min(rx2, x2), min(ry2, y2)), color, -1)
        elif shape_type == 'circle':
            radius = np.random.randint(max(1, sign_w // 8), max(2, sign_w // 3))
            cv2.circle(overlay, rand_point(), radius, color, -1 if np.random.random() > 0.5 else thickness)
        elif shape_type == 'polyline':
            pts = np.array([rand_point() for _ in range(np.random.randint(3, 7))], dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)
    alpha = np.random.uniform(0.55, 0.80)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def apply_illumination(image, bbox):
    """
    Simulates adverse lighting conditions on the sign by applying a randomised
    brightness/gamma shift to the bounding box region. Can produce either
    over-exposure (harsh sunlight) or under-exposure (night / shadow) effects.

    Args:
        image: BGR image (numpy array)
        bbox:  YOLO-format (x_center, y_center, width, height) normalized
    """
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox

    x1 = int((x_center - b_w / 2) * w)
    y1 = int((y_center - b_h / 2) * h)
    x2 = int((x_center + b_w / 2) * w)
    y2 = int((y_center + b_h / 2) * h)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    region = image[y1:y2, x1:x2].astype(np.float32)

    # Randomly choose overexposure or underexposure
    if np.random.random() > 0.5:
        # Overexposure: strong additive brightness + gamma < 1
        brightness = np.random.uniform(60, 120)
        gamma = np.random.uniform(0.4, 0.75)
        region = np.clip(region + brightness, 0, 255)
    else:
        # Underexposure: darken + gamma > 1
        brightness = np.random.uniform(-80, -30)
        gamma = np.random.uniform(1.4, 2.5)
        region = np.clip(region + brightness, 0, 255)

    # Apply gamma correction
    region = region / 255.0
    region = np.power(region, gamma)
    region = np.clip(region * 255, 0, 255).astype(np.uint8)

    # Soft feathered edge so the effect blends naturally
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=((x2 - x1) // 2, (y2 - y1) // 2),
        axes=((x2 - x1) // 2, (y2 - y1) // 2),
        angle=0, startAngle=0, endAngle=360,
        color=1.0, thickness=-1
    )
    mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., np.newaxis]

    original_region = image[y1:y2, x1:x2].astype(np.float32)
    blended = (mask * region + (1 - mask) * original_region).astype(np.uint8)
    image[y1:y2, x1:x2] = blended
    return image


# ── Dataset generation helpers ───────────────────────────────────────────────

# NOTE: IF YOU ADD A NEW ATTACK TYPE, ADD IT TO THIS MAP
ATTACK_MAP = {
    "tape":            apply_digital_tape,
    "patch":           apply_adversarial_patch,
    "heavy_occlusion": apply_heavy_occlusion,
    "graffiti":        apply_graffiti,
    "illumination":    apply_illumination,
}


def generate_attack_dataset(base_path, attack_type="patch"):
    """
    Applies the chosen attack to every image in valid/images and writes
    the results to attacked_val/images (labels are copied unchanged).

    attack_type options: 'tape', 'patch', 'heavy_occlusion', 'graffiti', 'illumination'
    """
    if attack_type not in ATTACK_MAP:
        print(f"❌ Invalid attack type: '{attack_type}'. Valid options: {list(ATTACK_MAP.keys())}")
        attack_func = apply_adversarial_patch
    else:
        attack_func = ATTACK_MAP[attack_type]

    val_images_path  = os.path.join(base_path, 'valid', 'images')
    val_labels_path  = os.path.join(base_path, 'valid', 'labels')
    output_base      = os.path.join(base_path, 'attacked_val')
    output_images    = os.path.join(output_base, 'images')
    output_labels    = os.path.join(output_base, 'labels')

    os.makedirs(output_images, exist_ok=True)

    if os.path.exists(val_labels_path):
        if os.path.exists(output_labels):
            shutil.rmtree(output_labels)
        shutil.copytree(val_labels_path, output_labels)
        print(f"✅ Labels synced for '{attack_type}' attack.")

    print(f"Creating '{attack_type}' attacked images...")
    for img_name in os.listdir(val_images_path):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(os.path.join(val_images_path, img_name))
        label_file = os.path.join(val_labels_path, img_name.rsplit('.', 1)[0] + '.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        bbox = [float(x) for x in parts[1:]]
                        img = attack_func(img, bbox)

        cv2.imwrite(os.path.join(output_images, img_name), img)

    print(f"✅ Done. Attacked images saved to {output_images}")


def create_attack_yaml(base_path, attack_type="patch"):
    """Generates attack.yaml pointing at the attacked_val split."""
    original_yaml = os.path.join(base_path, 'data.yaml')
    attack_yaml   = os.path.join(base_path, 'attack.yaml')

    if not os.path.exists(original_yaml):
        print(f"❌ data.yaml not found at {original_yaml}")
        return

    with open(original_yaml, 'r') as f:
        data = yaml.safe_load(f)

    attack_config = {
        'path':  base_path,
        'train': 'train/images',
        'val':   'attacked_val/images',
        'test':  'valid/images',
        'nc':    data['nc'],
        'names': data['names'],
    }

    with open(attack_yaml, 'w') as f:
        yaml.dump(attack_config, f)
    print(f"✅ Generated {attack_yaml}")