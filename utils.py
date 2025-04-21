import matplotlib.pyplot as plt
import numpy as np
import torch

# Функция денормализации
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Денормализует тензор изображения для визуализации."""
    if not isinstance(tensor, torch.Tensor):
         return tensor # Если это уже numpy array

    # Клонируем тензор
    tensor = tensor.clone().detach().cpu()

    # Применяем обратное преобразование для каждого канала
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # Преобразуем в numpy и меняем порядок осей (C, H, W) -> (H, W, C)
    img_np = tensor.numpy().transpose(1, 2, 0)

    # Обрезаем значения до [0, 1] и конвертируем в uint8 [0, 255]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def visualize_predictions(images, true_masks, predicted_masks, num_examples=5, filename="prediction_examples.png"):
    """Визуализирует N примеров: Изображение | Истинная маска | Предсказанная маска."""
    num_examples = min(num_examples, len(images))
    if num_examples == 0:
        print("Нет примеров для визуализации.")
        return

    plt.figure(figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        # Исходное изображение 
        img = denormalize(images[i])

        # Истинная маска (предполагаем (B, 1, H, W) тензор)
        true_mask = true_masks[i].cpu().numpy().squeeze() 

        # Предсказанная маска (предполагаем (B, 1, H, W) тензор вероятностей)
        pred_mask_prob = predicted_masks[i].cpu().numpy().squeeze()
        pred_mask_binary = (pred_mask_prob > 0.5).astype(np.uint8) # Бинаризуем по порогу 0.5

        # Отображение
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f"Изображение {i+1}")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title(f"Истинная маска {i+1}")
        plt.axis('off')

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(pred_mask_binary, cmap='gray')
        plt.title(f"Предсказание {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Визуализация предсказаний сохранена в файл: {filename}")
    plt.show() 
    plt.close()