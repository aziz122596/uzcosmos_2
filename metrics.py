import torch

SMOOTH = 1e-6 # для избежания деления на ноль

def iou_score(output, target, threshold=0.5):
    """Вычисляет IoU (Intersection over Union) для бинарной сегментации.
       Предполагает, что output - это вероятности (после sigmoid).
    """
    with torch.no_grad():
        pred = (output > threshold).float()
        target = target.float()

        intersection = (pred * target).sum(dim=(1, 2, 3)) # Суммируем по H, W, C (C=1)
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + SMOOTH) / (union + SMOOTH) # Добавляем smooth
    return iou.mean() # Усредняем по батчу

def dice_coeff(output, target, threshold=0.5):
    """Вычисляет Dice Coefficient (F1-score) для бинарной сегментации.
       Предполагает, что output - это вероятности (после sigmoid).
    """
    with torch.no_grad():
        pred = (output > threshold).float()
        target = target.float()

        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2. * intersection + SMOOTH) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + SMOOTH)
    return dice.mean() # Усредняем по батчу