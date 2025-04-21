import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split 
from dataset import RoadDataset, get_validation_augmentation, get_preprocessing
from model import build_model
from metrics import iou_score, dice_coeff
from utils import visualize_predictions # Импортируем функцию визуализации

# --- Конфигурация ---
# путь в датасету
DATA_DIR = '/path/dataset/' # Путь к датасету
IMAGE_DIR_NAME = 'images'
MASK_DIR_NAME = 'masks'
IMG_EXTENSION = '.tiff'
MASK_EXTENSION = '.tif'

ENCODER = 'resnet34'
ENCODER_WEIGHTS = None 
NUM_CLASSES = 1
ACTIVATION = 'sigmoid'

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = './checkpoints_road_seg/best_model.pth' 
TRAIN_VAL_SPLIT = 0.8 
RANDOM_STATE = 42      
NUM_VIS_EXAMPLES = 5   

print(f"Используемое устройство: {DEVICE}")

# --- Подготовка тестовых данных ---
# В проекте иметь отдельный test set.
print("Подготовка тестовых (валидационных) данных...")
image_dir_full = os.path.join(DATA_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir_full) or not os.path.isdir(mask_dir_full):
     print(f"Ошибка: Папки с данными не найдены: {image_dir_full}, {mask_dir_full}")
     exit()

try:
    all_ids = sorted([f.split('.')[0] for f in os.listdir(image_dir_full) if f.endswith(IMG_EXTENSION)])
    all_ids = [id_ for id_ in all_ids if os.path.exists(os.path.join(mask_dir_full, id_ + MASK_EXTENSION))]
except FileNotFoundError:
     print(f"Ошибка чтения файлов из папок данных.")
     exit()


if not all_ids:
    print("Ошибка: Не найдены пары изображений/масок для оценки.")
    exit()

_, test_ids = train_test_split(all_ids, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Используется {len(test_ids)} примеров для оценки (валидационный набор).")

try:
    preprocessing_fn_eval = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet' if ENCODER_WEIGHTS == 'imagenet' else None)
except:
    preprocessing_fn_eval = None

# Создание тестового датасета и загрузчика
test_dataset = RoadDataset(
    image_dir_full, mask_dir_full, test_ids, IMG_EXTENSION, MASK_EXTENSION,
    transform=get_validation_augmentation(IMG_HEIGHT, IMG_WIDTH), 
    preprocessing=get_preprocessing(preprocessing_fn_eval)
)

num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
num_workers = min(num_workers, 4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
print("Тестовый датасет и загрузчик созданы.")


# --- Загрузка модели ---
print(f"Загрузка лучшей модели из {CHECKPOINT_PATH}...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Ошибка: Файл модели не найден: {CHECKPOINT_PATH}")
    print("Убедитесь, что скрипт train.py был запущен и модель сохранена.")
    exit()

model = build_model(ENCODER, encoder_weights=None, num_classes=NUM_CLASSES, activation=ACTIVATION) # Веса загрузим из файла
try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() 
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке весов модели: {e}")
    exit()

# --- Оценка на тестовом наборе ---
print("Запуск оценки на тестовом наборе...")
test_iou_total = 0.0
test_dice_total = 0.0
all_images_for_vis = []
all_true_masks_for_vis = []
all_pred_masks_for_vis = []

with torch.no_grad():
    test_loop = tqdm(test_loader, desc="Testing", leave=False)
    for i, (images, masks) in enumerate(test_loop):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True) 

        outputs = model(images) # outputs shape: (B, 1, H, W), float32 (вероятности)

        # Вычисляем метрики на батче
        batch_iou = iou_score(outputs, masks)
        batch_dice = dice_coeff(outputs, masks)
        test_iou_total += batch_iou.item()
        test_dice_total += batch_dice.item()

        # Сохраняем несколько первых примеров для визуализации
        if i == 0 and len(all_images_for_vis) < NUM_VIS_EXAMPLES :
            all_images_for_vis.append(images.cpu())
            all_true_masks_for_vis.append(masks.cpu())
            all_pred_masks_for_vis.append(outputs.cpu())

        test_loop.set_postfix(iou=f"{batch_iou:.4f}", dice=f"{batch_dice:.4f}")

# Рассчитываем средние метрики по всем батчам
final_test_iou = test_iou_total / len(test_loader)
final_test_dice = test_dice_total / len(test_loader)

print("\n--- Результаты на тестовом (валидационном) наборе ---")
print(f"Test IoU: {final_test_iou:.4f}")
print(f"Test Dice: {final_test_dice:.4f}")

# --- Визуализация результатов ---
print("\nСоздание визуализации предсказаний...")
if all_images_for_vis:
    vis_images = torch.cat(all_images_for_vis)
    vis_true_masks = torch.cat(all_true_masks_for_vis)
    vis_pred_masks = torch.cat(all_pred_masks_for_vis)

    visualize_predictions(
        vis_images,
        vis_true_masks,
        vis_pred_masks,
        num_examples=NUM_VIS_EXAMPLES,
        filename="prediction_examples_road_seg.png"
    )
else:
    print("Не удалось собрать примеры для визуализации.")

print("\n--- Скрипт evaluate.py завершен ---")