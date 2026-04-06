"""
Expanded 70/30 training script — local GPU version.
Mirrors notebooks/expanded_70_30_train.ipynb without any Colab dependencies.

Usage:
    python src/train_expanded_70_30.py

Configure the three variables in the CONFIG section below before running.
"""

import os
import random
import shutil

import cv2
import numpy as np
import yaml
import albumentations as A
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(ROOT, 'datasets', 'archive')
MODEL_IN  = os.path.join(ROOT, 'runs', 'detect', 'yolov8_aug_graffiti9', 'weights', 'last.pt')

# ── CONFIG — change these before running ─────────────────────────────────────
ATTACK_TYPE = 'graffiti'   # 'tape', 'patch', 'graffiti', 'illumination'
SPLIT_RATIO = 0.7          # fraction of training images that get attacked
EPOCHS      = 50
BATCH       = 64
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FOLDER = f'train_aug_{ATTACK_TYPE}'
ROBUST_YAML  = f'robust_aug_{ATTACK_TYPE}.yaml'
ATTACK_YAML  = f'attack_aug_{ATTACK_TYPE}.yaml'
RUN_NAME     = f'yolov8_aug_{ATTACK_TYPE}'


# ── Attack functions ──────────────────────────────────────────────────────────

def apply_digital_tape(image, bbox, color=(50, 50, 50), thickness_ratio=0.2):
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox
    x1 = int((x_center - b_w / 2) * w)
    x2 = int((x_center + b_w / 2) * w)
    y1 = int((y_center - b_h / 2) * h)
    y2 = int((y_center + b_h / 2) * h)
    tape_h  = int((y2 - y1) * thickness_ratio)
    tape_y1 = int(y_center * h - tape_h / 2)
    tape_y2 = int(y_center * h + tape_h / 2)
    cv2.rectangle(image, (x1, tape_y1), (x2, tape_y2), color, -1)
    return image


def apply_adversarial_patch(image, bbox):
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox
    x1 = int((x_center - b_w / 4) * w)
    y1 = int((y_center - b_h / 4) * h)
    x2 = int((x_center + b_w / 4) * w)
    y2 = int((y_center + b_h / 4) * h)
    patch_h = max(1, y2 - y1)
    patch_w = max(1, x2 - x1)
    image[y1:y2, x1:x2] = np.random.randint(0, 255, (patch_h, patch_w, 3), dtype=np.uint8)
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
    h, w, _ = image.shape
    x_center, y_center, b_w, b_h = bbox
    x1, y1 = max(0, int((x_center - b_w / 2) * w)), max(0, int((y_center - b_h / 2) * h))
    x2, y2 = min(w, int((x_center + b_w / 2) * w)), min(h, int((y_center + b_h / 2) * h))
    if x2 <= x1 or y2 <= y1:
        return image
    region = image[y1:y2, x1:x2].astype(np.float32)
    if np.random.random() > 0.5:
        region = np.clip(region + np.random.uniform(60, 120), 0, 255)
        gamma  = np.random.uniform(0.4, 0.75)
    else:
        region = np.clip(region + np.random.uniform(-80, -30), 0, 255)
        gamma  = np.random.uniform(1.4, 2.5)
    region = np.clip(np.power(region / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    mask   = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    cv2.ellipse(mask, ((x2-x1)//2, (y2-y1)//2), ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 1.0, -1)
    mask   = cv2.GaussianBlur(mask, (21, 21), 0)[..., np.newaxis]
    orig   = image[y1:y2, x1:x2].astype(np.float32)
    image[y1:y2, x1:x2] = (mask * region + (1 - mask) * orig).astype(np.uint8)
    return image


ATTACK_MAP = {
    'tape':         apply_digital_tape,
    'patch':        apply_adversarial_patch,
    'graffiti':     apply_graffiti,
    'illumination': apply_illumination,
}

# ── Augmentation pipeline ─────────────────────────────────────────────────────

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.ImageCompression(quality_range=(75, 100), p=0.3),
    A.RandomShadow(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_label_file(label_path):
    class_ids, bboxes = [], []
    if not os.path.exists(label_path):
        return class_ids, bboxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes


# ── Step 1: Build expanded training set ──────────────────────────────────────

def build_expanded_train_set(base_path, train_folder, attack_type, split_ratio=0.7):
    train_images_src = os.path.join(base_path, 'train', 'images')
    train_labels_src = os.path.join(base_path, 'train', 'labels')
    robust_images    = os.path.join(base_path, train_folder, 'images')
    robust_labels    = os.path.join(base_path, train_folder, 'labels')
    os.makedirs(robust_images, exist_ok=True)
    os.makedirs(robust_labels, exist_ok=True)

    attack_func = ATTACK_MAP[attack_type]
    all_images  = [f for f in os.listdir(train_images_src) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)
    attack_set  = set(all_images[:int(len(all_images) * split_ratio)])

    print(f"Original images      : {len(all_images)}")
    print(f"  Attacked ({int(split_ratio*100)}%)    : {len(attack_set)}")
    print(f"  Clean    ({int((1-split_ratio)*100)}%)    : {len(all_images) - len(attack_set)}")
    print(f"  + Augmented copies  : {len(all_images)}")
    print(f"  Effective total     : ~{len(all_images) * 2}")

    skipped = 0
    for i, img_name in enumerate(all_images):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i+1}/{len(all_images)}...")

        stem    = img_name.rsplit('.', 1)[0]
        src_img = os.path.join(train_images_src, img_name)
        src_lbl = os.path.join(train_labels_src, stem + '.txt')

        img = cv2.imread(src_img)
        if img is None or not os.path.exists(src_lbl):
            skipped += 1
            continue

        class_ids, bboxes = parse_label_file(src_lbl)

        if img_name in attack_set:
            for bbox in bboxes:
                img = attack_func(img, bbox)

        cv2.imwrite(os.path.join(robust_images, img_name), img)
        shutil.copy(src_lbl, os.path.join(robust_labels, stem + '.txt'))

        try:
            if bboxes:
                result     = augment(image=img, bboxes=bboxes, class_labels=class_ids)
                aug_img    = result['image']
                aug_bboxes = result['bboxes']
                aug_labels = result['class_labels']
            else:
                aug_img, aug_bboxes, aug_labels = img, [], []
        except Exception:
            aug_img, aug_bboxes, aug_labels = img, bboxes, class_ids

        cv2.imwrite(os.path.join(robust_images, stem + '_aug.jpg'), aug_img)
        with open(os.path.join(robust_labels, stem + '_aug.txt'), 'w') as f:
            for cls, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")

    print(f"\nExpanded training set created at: {os.path.join(base_path, train_folder)}")
    if skipped:
        print(f"Skipped {skipped} images (unreadable or missing labels).")


# ── Step 2: Generate YAMLs ────────────────────────────────────────────────────

def write_yamls(base_path, train_folder, robust_yaml, attack_yaml):
    with open(os.path.join(ROOT, 'data.yaml'), 'r') as f:
        ref = yaml.safe_load(f)
    class_names = ref['names']

    with open(os.path.join(base_path, robust_yaml), 'w') as f:
        yaml.dump({
            'path':  base_path,
            'train': f'{train_folder}/images',
            'val':   'valid/images',
            'test':  'valid/images',
            'nc':    264,
            'names': class_names,
        }, f)
    print(f"Written: {robust_yaml}")

    with open(os.path.join(base_path, attack_yaml), 'w') as f:
        yaml.dump({
            'path':  base_path,
            'train': f'{train_folder}/images',
            'val':   'attacked_val/images',
            'test':  'attacked_val/images',
            'nc':    264,
            'names': class_names,
        }, f)
    print(f"Written: {attack_yaml}")


# ── Step 3: Build attacked validation set ────────────────────────────────────

def create_attacked_val_set(base_path, attack_type):
    val_images_src = os.path.join(base_path, 'valid', 'images')
    val_labels_src = os.path.join(base_path, 'valid', 'labels')
    attacked_root  = os.path.join(base_path, 'attacked_val')

    if os.path.exists(attacked_root):
        shutil.rmtree(attacked_root)
    os.makedirs(os.path.join(attacked_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(attacked_root, 'labels'), exist_ok=True)

    attack_func = ATTACK_MAP[attack_type]
    all_images  = [f for f in os.listdir(val_images_src) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Attacking {len(all_images)} validation images with '{attack_type}'...")

    for img_name in all_images:
        img      = cv2.imread(os.path.join(val_images_src, img_name))
        stem     = img_name.rsplit('.', 1)[0]
        src_lbl  = os.path.join(val_labels_src, stem + '.txt')
        dst_lbl  = os.path.join(attacked_root, 'labels', stem + '.txt')

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            _, bboxes = parse_label_file(src_lbl)
            for bbox in bboxes:
                img = attack_func(img, bbox)

        cv2.imwrite(os.path.join(attacked_root, 'images', img_name), img)

    print(f"Attacked validation set created at {attacked_root}")


# ── Step 4: Train ─────────────────────────────────────────────────────────────

def train(base_path, robust_yaml, run_name, model_in, epochs, batch):
    model = YOLO(model_in)
    model.train(
        data=os.path.join(base_path, robust_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        lr0=0.001,
        mosaic=0.5,
        mixup=0.05,
        device=0,
        name=run_name,
        workers=2,
        resume=False
    )
    return os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────

def evaluate(model_path, base_path, robust_yaml, attack_yaml, attack_type):
    model = YOLO(model_path)

    print('\n--- CLEAN VALIDATION DATA ---')
    clean_results = model.val(data=os.path.join(base_path, robust_yaml))

    print(f'\n--- {attack_type.upper()} ATTACKED VALIDATION DATA ---')
    attack_results = model.val(data=os.path.join(base_path, attack_yaml))

    print('\n' + '='*65)
    print(f"{'Metric':<20} {'Clean Val':>18} {f'{attack_type} Attack':>18}")
    print('-' * 65)
    for label, key in zip(
        ['mAP50', 'mAP50-95', 'Precision', 'Recall'],
        ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
    ):
        c = clean_results.results_dict.get(key, float('nan'))
        a = attack_results.results_dict.get(key, float('nan'))
        print(f"{label:<20} {c:>18.4f} {a:>18.4f}")
    print('=' * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Attack type  : {ATTACK_TYPE}")
    print(f"Split ratio  : {int(SPLIT_RATIO*100)}% attacked / {int((1-SPLIT_RATIO)*100)}% clean")
    print(f"Base model   : {MODEL_IN}")
    print(f"Dataset      : {BASE_PATH}\n")

    print("=== Step 1: Building expanded training set ===")
    build_expanded_train_set(BASE_PATH, TRAIN_FOLDER, ATTACK_TYPE, SPLIT_RATIO)

    print("\n=== Step 2: Writing YAMLs ===")
    write_yamls(BASE_PATH, TRAIN_FOLDER, ROBUST_YAML, ATTACK_YAML)

    print("\n=== Step 3: Building attacked validation set ===")
    create_attacked_val_set(BASE_PATH, ATTACK_TYPE)

    print("\n=== Step 4: Training ===")
    trained_model_path = train(BASE_PATH, ROBUST_YAML, RUN_NAME, MODEL_IN, EPOCHS, BATCH)

    print(f"\n=== Step 5: Evaluating — model at {trained_model_path} ===")
    evaluate(trained_model_path, BASE_PATH, ROBUST_YAML, ATTACK_YAML, ATTACK_TYPE)

    print(f"\nDone. Trained model saved to: {trained_model_path}")
