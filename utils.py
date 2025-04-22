import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path="training_history.png"):
    """
    Строит графики истории обучения модели Keras (loss и метрики).
    """
    # Проверяем наличие ключей 
    loss_key = 'loss'
    val_loss_key = 'val_loss'
    metrics_keys = [k for k in history.history.keys() if k not in ['loss', 'val_loss', 'lr']]
    val_metrics_keys = [k for k in metrics_keys if k.startswith('val_')]
    train_metrics_keys = [k for k in metrics_keys if not k.startswith('val_')]

    num_plots = 1 + len(train_metrics_keys) 
    plt.figure(figsize=(6 * num_plots, 5))

    # График Лосса
    ax_loss = plt.subplot(1, num_plots, 1)
    ax_loss.plot(history.epoch, history.history[loss_key], label="Train loss")
    if val_loss_key in history.history:
        ax_loss.plot(history.epoch, history.history[val_loss_key], label="Validation loss")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("История функции потерь")
    ax_loss.legend()
    ax_loss.grid(True)

    # Графики Метрик
    for i, metric_name in enumerate(train_metrics_keys):
        val_metric_name = f"val_{metric_name}"
        ax_acc = plt.subplot(1, num_plots, i + 2)
        ax_acc.plot(history.epoch, history.history[metric_name], label=f"Train {metric_name}")
        if val_metric_name in history.history:
            ax_acc.plot(history.epoch, history.history[val_metric_name], label=f"Validation {metric_name}")
        ax_acc.set_xlabel("Эпоха")
        ax_acc.set_ylabel("Метрика")
        ax_acc.set_title(f"История метрики: {metric_name}")
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"График истории обучения сохранен в: {save_path}")
    plt.close()


def visualize_predictions(image, true_mask, predicted_mask, save_path="prediction_example.png"):
    """
    Визуализирует один пример: Изображение | Истинная маска | Предсказанная маска.
    Предполагает, что image - это (H, W, C) numpy array в диапазоне [0, 1] или [0, 255],
    а маски - (H, W) или (H, W, 1) бинарные numpy array (0/1).
    """
    # Нормализуем изображение к [0, 1] для отображения, если оно [0, 255]
    if image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0, 1) 

    # Убираем размерность канала у масок, если есть
    if true_mask.ndim == 3:
        true_mask = true_mask.squeeze(-1)
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask.squeeze(-1)

    # Бинаризуем предсказанную маску 
    if predicted_mask.max() <= 1.0 and predicted_mask.min() >= 0.0 and len(np.unique(predicted_mask)) > 2: 
         predicted_mask = (predicted_mask > 0.5).astype(np.uint8) # Порог 0.5

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image)
    axs[0].set_title("Изображение")
    axs[0].axis('off')

    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("Истинная маска")
    axs[1].axis('off')

    axs[2].imshow(predicted_mask, cmap="gray")
    axs[2].set_title("Предсказанная маска")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Визуализация примера сохранена в: {save_path}")
    plt.close()