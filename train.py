import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
from dataset import RoadDataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
from model import build_model
from metrics import iou_score, dice_coeff

# --- Конфигурация ---
# необходимо указать путь к папке с данными
DATA_DIR = '/data/massachusetts_roads/' 
IMAGE_DIR_NAME = 'images'  
MASK_DIR_NAME = 'masks'    
IMG_EXTENSION = '.tiff'    
MASK_EXTENSION = '.tif'   

# Параметры модели
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet' # 'imagenet' для использования предобученных весов 
NUM_CLASSES = 1             # 1 для бинарной сегментации (дорога/не дорога)
ACTIVATION = 'sigmoid'    

# Параметры обучения
IMG_HEIGHT = 256            # Высота изображений для обучения
IMG_WIDTH = 256             # Ширина изображений для обучения
BATCH_SIZE = 8             
EPOCHS = 25                 
LEARNING_RATE = 1e-4        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = './checkpoints_road_seg'
TRAIN_VAL_SPLIT = 0.8      
RANDOM_STATE = 42           

# Создаем папку для чекпоинтов, если ее нет
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Используемое устройство: {DEVICE}")

# --- Подготовка данных ---
print("Подготовка данных...")
image_dir_full = os.path.join(DATA_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_DIR, MASK_DIR_NAME)

# Проверка существования папок
if not os.path.isdir(image_dir_full):
    print(f"Ошибка: Папка с изображениями не найдена: {image_dir_full}")
    exit()
if not os.path.isdir(mask_dir_full):
    print(f"Ошибка: Папка с масками не найдена: {mask_dir_full}")
    exit()


# Получаем список ID файлов 
try:
    all_ids = sorted([f.split('.')[0] for f in os.listdir(image_dir_full) if f.endswith(IMG_EXTENSION)])
    # Убедимся, что для каждого изображения есть маска
    all_ids = [id_ for id_ in all_ids if os.path.exists(os.path.join(mask_dir_full, id_ + MASK_EXTENSION))]
except FileNotFoundError:
     print(f"Ошибка: Не удалось прочитать файлы из {image_dir_full} или {mask_dir_full}.")
     exit()


if not all_ids:
    print(f"Ошибка: Не найдены совпадающие пары изображений ({IMG_EXTENSION}) / масок ({MASK_EXTENSION})")
    print(f"в папках {image_dir_full} и {mask_dir_full}.")
    exit()

print(f"Найдено {len(all_ids)} пар изображений/масок.")

# Разделение на train/validation
train_ids, val_ids = train_test_split(all_ids, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Разделение данных: {len(train_ids)} на обучение, {len(val_ids)} на валидацию.")

# Получаем функцию препроцессинга из SMP 
try:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    print(f"Используется функция препроцессинга для энкодера {ENCODER} с весами {ENCODER_WEIGHTS}.")
except:
    print(f"Предупреждение: Не удалось получить функцию препроцессинга для {ENCODER}/{ENCODER_WEIGHTS}. Используется стандартная нормализация.")
    preprocessing_fn = None

# Создание датасетов и загрузчиков
train_dataset = RoadDataset(
    image_dir_full, mask_dir_full, train_ids, IMG_EXTENSION, MASK_EXTENSION,
    transform=get_training_augmentation(IMG_HEIGHT, IMG_WIDTH),
    preprocessing=get_preprocessing(preprocessing_fn)
)
val_dataset = RoadDataset(
    image_dir_full, mask_dir_full, val_ids, IMG_EXTENSION, MASK_EXTENSION,
    transform=get_validation_augmentation(IMG_HEIGHT, IMG_WIDTH), 
    preprocessing=get_preprocessing(preprocessing_fn)
)

num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
num_workers = min(num_workers, 4) 
print(f"Используется {num_workers} воркеров для DataLoader.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
print("Датасеты и загрузчики созданы.")

# --- Модель, Лосс, Оптимизатор ---
print("Создание модели, лосса и оптимизатора...")
model = build_model(ENCODER, ENCODER_WEIGHTS, NUM_CLASSES, ACTIVATION)
model.to(DEVICE)

# Выбор функции потерь 
loss_fn = smp.losses.DiceLoss(mode='binary')


# Оптимизатор 
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Метрики для отслеживания 
metrics_dict = {'iou': iou_score, 'dice': dice_coeff}
print("Модель, лосс и оптимизатор настроены.")


# --- Цикл обучения и валидации ---
history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
best_val_metric = 0.0 
metric_to_monitor = 'dice' 

print(f"Начало обучения на {EPOCHS} эпох...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\n--- Эпоха {epoch+1}/{EPOCHS} ---")

    # --- Фаза обучения ---
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for images, masks in train_loop:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)


        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_loop.set_postfix(loss=f"{loss.item():.4f}")

    epoch_train_loss = running_loss / len(train_loader.dataset)
    history['train_loss'].append(epoch_train_loss)
    print(f"Train Loss: {epoch_train_loss:.4f}")

    # --- Фаза валидации ---
    model.eval()
    val_loss = 0.0
    val_metrics_accum = {name: 0.0 for name in metrics_dict} # Аккумулятор для метрик

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Eval Epoch {epoch+1}", leave=False)
        for images, masks in val_loop:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images) # Получаем вероятности
            loss = loss_fn(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Вычисляем метрики на батче
            for name, metric_fn in metrics_dict.items():
                batch_metric = metric_fn(outputs, masks) 
                val_metrics_accum[name] += batch_metric.item() * images.size(0)

            # Формируем строку для лога tqdm
            metrics_str = ", ".join([f"{name}={val_metrics_accum[name]/((val_loop.n + 1)*images.size(0)):.4f}" for name in metrics_dict])
            val_loop.set_postfix(loss=f"{loss.item():.4f}", metrics=metrics_str)


    # Рассчитываем средние значения за эпоху
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_metrics = {name: val_metrics_accum[name] / len(val_loader.dataset) for name in metrics_dict}

    history['val_loss'].append(epoch_val_loss)
    for name, value in epoch_val_metrics.items():
         history[f'val_{name}'].append(value) 

    metrics_log_str = ", ".join([f"Val {name.capitalize()}: {value:.4f}" for name, value in epoch_val_metrics.items()])
    print(f"Val Loss: {epoch_val_loss:.4f}, {metrics_log_str}")

    scheduler.step(epoch_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Текущий Learning Rate: {current_lr:.6f}")


    # Сохраняем лучшую модель по выбранной метрике 
    current_metric_value = epoch_val_metrics[metric_to_monitor]
    if current_metric_value > best_val_metric:
        best_val_metric = current_metric_value
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"!!! Модель сохранена как лучшая (по {metric_to_monitor}={best_val_metric:.4f}) в {checkpoint_path} !!!")

    epoch_end_time = time.time()
    print(f"Эпоха {epoch+1} завершена за {epoch_end_time - epoch_start_time:.2f} секунд.")


# --- Завершение обучения ---
total_training_time = time.time() - start_time
print(f"\nОбучение завершено за {total_training_time / 60:.2f} минут.")
print(f"Лучшее значение метрики валидации ({metric_to_monitor}): {best_val_metric:.4f}")

# --- Построение графиков обучения ---
print("Построение графиков обучения...")
plt.figure(figsize=(12, 5))

# График Лосса
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss')
plt.plot(range(1, EPOCHS + 1), history['val_loss'], label='Validation Loss')
plt.title('История функции потерь')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# График Метрик (IoU и Dice)
plt.subplot(1, 2, 2)
if 'val_iou' in history:
    plt.plot(range(1, EPOCHS + 1), history['val_iou'], label='Validation IoU')
if 'val_dice' in history:
    plt.plot(range(1, EPOCHS + 1), history['val_dice'], label='Validation Dice')
plt.title('История метрик валидации')
plt.xlabel('Эпоха')
plt.ylabel('Значение метрики')
if 'val_iou' in history or 'val_dice' in history:
    plt.legend()
plt.grid(True)

plt.tight_layout()
history_plot_path = 'training_history_road_seg.png'
plt.savefig(history_plot_path)
print(f"Графики обучения сохранены в файл: {history_plot_path}")
# plt.show()
plt.close()

print("\n--- Скрипт train.py завершен ---")