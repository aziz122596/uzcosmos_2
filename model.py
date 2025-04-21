import torch
import segmentation_models_pytorch as smp

def build_model(encoder='resnet34', encoder_weights='imagenet', num_classes=1, activation='sigmoid'):
    """
    Создает модель сегментации U-Net с использованием segmentation-models-pytorch.

    Args:
        encoder (str): Имя энкодера (бэкбона).
        encoder_weights (str): Веса для инициализации энкодера ('imagenet' или None).
        num_classes (int): Количество классов (1 для бинарной сегментации).
        activation (str): Функция активации для выходного слоя ('sigmoid' для бинарной, 'softmax' для мультиклассовой, None если лосс сам применяет активацию).

    Returns:
        torch.nn.Module: Модель сегментации.
    """
    print(f"Создание модели U-Net с энкодером {encoder}...")
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,            
        classes=num_classes,      
        activation=activation,   
    )
    return model
