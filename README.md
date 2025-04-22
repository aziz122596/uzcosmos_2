# Сегментация дорог на аэроснимках (Keras + Segmentation Models)

Этот проект реализует модель семантической сегментации (U-Net с бэкбоном EfficientNet/ResNet) с использованием Keras и библиотеки `segmentation-models` для обнаружения дорог на аэроснимках.

## Структура проекта

- `train.py`: Основной скрипт для обучения и валидации модели.
- `evaluate.py`: Скрипт для оценки обученной модели на тестовом наборе и визуализации предсказаний.
- `main_1.py`: Содержит класс `SegmentationDataGenerator` (Keras Sequence) для загрузки данных и функции для аугментации/препроцессинга.
- `model.py`: Содержит функцию `build_unet_model` для создания архитектуры сегментации (U-Net).
- `metrics_and_losses.py`: Определяет или импортирует функции потерь и метрик из `segmentation_models`.
- `utils.py`: Содержит вспомогательные функции (построение графиков, визуализация).
- `requirements.txt`: Список необходимых Python библиотек (для Keras/TensorFlow).
- `README.md`: Этот файл.
- `.gitignore`: Определяет файлы и папки, игнорируемые Git.
- `checkpoints_keras/`: Папка (создается `train.py`) для сохранения лучшей модели (`.h5` файлы).
- `logs_keras/`: Папка (создается `train.py`) для логов TensorBoard.
- `training_history_keras.png`: Графики лосса/метрик обучения (создается `train.py`).
- `prediction_example_*.png`: Примеры визуализации предсказаний (создается `evaluate.py`).

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
   git clone [https://github.com/aziz122596/uzcosmos_1.git] (https://github.com/aziz122596/uzcosmos_1.git) 
    cd uzcosmos_2
    ```

2.  **Скачайте датасет:**
    - Выберите и скачайте один из датасетов: [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/) или [DeepGlobe Road Extraction Dataset](https://competitions.codalab.org/competitions/18467). import kagglehub

# Download latest version
path = kagglehub.dataset_download("insaff/massachusetts-roads-dataset")

print("Path to dataset files:", path)
    - Распакуйте архив.
    - **ВАЖНО:** Убедитесь, что у вас есть папка (например, `training`), которая содержит подпапки `input` (с `.png` изображениями) и `output` (с `.png` масками). Запомните **полный путь** к этой папке (`training` или аналогичной).

3.  **Настройте пути в скриптах:**
    - Откройте файлы `train.py` и `evaluate.py`.
    - Найдите переменную `DATA_ROOT_DIR` и **замените** `/path/to/your/training_data_folder/` на **реальный путь** к вашей папке, содержащей подпапки `input` и `output`.
    - Убедитесь, что `IMAGE_DIR_NAME = 'input'` и `MASK_DIR_NAME = 'output'`.
    - В `evaluate.py` найдите `MODEL_PATH` и подготовьтесь указать путь к файлу `.h5` лучшей модели после обучения.

4.  **Создайте и активируйте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # venv\Scripts\activate
    ```

5.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *Примечание: Установка `tensorflow` может потребовать специфических шагов для поддержки GPU (CUDA, cuDNN). См. официальную документацию TensorFlow.*

## Использование

1.  **Обучение модели:**
    Убедитесь, что путь `DATA_ROOT_DIR` в `train.py` настроен правильно. Запустите обучение:
    ```bash
    python train.py
    ```
    - Лучшая модель будет сохранена в папку `checkpoints_keras/` (имя файла будет включать эпоху и метрику).
    - Графики обучения будут сохранены в `training_history_keras.png`.
    - Логи для TensorBoard будут в папке `logs_keras/`.

2.  **Оценка модели:**
    - **Найдите имя лучшей модели** (`.h5` файл) в папке `checkpoints_keras/`.
    - **Укажите этот полный путь** в переменной `MODEL_PATH` в файле `evaluate.py`.
    - Убедитесь, что `DATA_ROOT_DIR` в `evaluate.py` указан верно.
    - Запустите оценку:
    ```bash
    python evaluate.py
    ```
    - Метрики (Loss, IoU, Dice) будут выведены в консоль.
    - Примеры предсказаний будут сохранены в файлы `prediction_example_*.png`.

## Подход
(Остается как в предыдущей версии README)
...

## Результаты (Пример)
(Остается как в предыдущей версии README)
...

## Возможные улучшения
(Остается как в предыдущей версии README)
...