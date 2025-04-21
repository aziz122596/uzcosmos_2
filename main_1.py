import os
import cv2 
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image 

# --- Функции Аугментации и Препроцессинга ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(height, width):
    """Возвращает конвейер аугментаций для обучающих данных."""
    train_transform = [
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussNoise(p=0.2),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(height, width):
    """Возвращает конвейер аугментаций для валидационных/тестовых данных."""
    test_transform = [
        A.Resize(height, width),
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn=None):
    """Возвращает конвейер препроцессинга (нормализация + конвертация в тензор).
    Args:
        preprocessing_fn (callable, optional): Функция нормализации, из segmentation_models_pytorch.
                                               Если None, используется простая нормализация делением на 255.
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    else:
        # Простая нормализация к [0, 1]
        _transform.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))

    _transform.append(ToTensorV2()) 
    return A.Compose(_transform)

# --- Класс Датасета ---
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_ids, img_ext, mask_ext, transform=None, preprocessing=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.preprocessing = preprocessing
        print(f"Dataset создан: {len(self.image_ids)} ID найдено.")
        if not self.image_ids:
             print(f"Предупреждение: Список image_ids пуст!")


    def __len__(self):
        return len(self.image_ids)

    def load_image(self, path):
        """Загружает изображение, пробуя OpenCV и PIL (для TIFF)."""
        try:
            # Пробуем OpenCV
            img = cv2.imread(path)
            if img is None: 
                raise ValueError("cv2.imread вернул None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            return img
        except Exception as e_cv2:
            try:
                img_pil = Image.open(path)
                img = np.array(img_pil)
                if len(img.shape) == 2: 
                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4: 
                     img = img[:, :, :3]
                return img
            except Exception as e_pil:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить изображение {path} ни через OpenCV, ни через PIL: {e_pil}")
                return None 

    def load_mask(self, path):
        """Загружает маску как одноканальное изображение."""
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                 raise ValueError("cv2.imread вернул None")
            return mask
        except Exception as e_cv2:

             try:
                # Пробуем PIL
                mask_pil = Image.open(path).convert('L') # Конвертируем в Grayscale
                mask = np.array(mask_pil)
                return mask
             except Exception as e_pil:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить маску {path} ни через OpenCV, ни через PIL: {e_pil}")
                return None


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id + self.img_ext)
        mask_path = os.path.join(self.mask_dir, image_id + self.mask_ext)

        image = self.load_image(img_path)
        mask = self.load_mask(mask_path)

        # Обработка ошибок загрузки
        if image is None or mask is None:
             print(f"Пропуск примера из-за ошибки загрузки: ID {image_id}")
             dummy_img = torch.zeros((3, 256, 256), dtype=torch.float32)
             dummy_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
             return dummy_img, dummy_mask


        # Бинаризация маски: все что не 0, становится 1
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        # 1. Применяем аугментации (если есть)
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                 print(f"Ошибка аугментации для ID {image_id}: {e}")
                 pass 

        # 2. Применяем препроцессинг (нормализация, ToTensorV2)
        if self.preprocessing:
            try:
                preprocessed = self.preprocessing(image=image, mask=mask)
                image = preprocessed['image']
                mask = preprocessed['mask'] 
            except Exception as e:
                 print(f"Ошибка препроцессинга для ID {image_id}: {e}")
                 dummy_img = torch.zeros((3, 256, 256), dtype=torch.float32)
                 dummy_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
                 return dummy_img, dummy_mask



        mask = mask.float()

        return image, mask