# Detection of Diabetic Retinopathy

Минимальный каркас проекта для дипломной работы по распознаванию диабетической ретинопатии и дальнейшего добавления explainable AI через `Grad-CAM`.

## Структура проекта

```text
Detection of Diabetic Retinopathy/
├── .vscode/
├── configs/
│   ├── default.yaml
│   ├── densenet121_baseline.yaml
│   ├── efficientnet_b0_baseline.yaml
│   ├── efficientnet_b0_multiclass.yaml
│   ├── efficientnet_b0_multiclass_crop.yaml
│   ├── efficientnet_b0_multiclass_crop_clahe.yaml
│   ├── resnet18_preprocessed.yaml
│   └── custom_cnn_scratch.yaml
├── data/
│   ├── raw/              # исходный датасет
│   └── processed/        # подготовленные данные и сплиты
├── models/               # сохранённые веса
├── notebooks/            # EDA и эксперименты
│   ├── comparison/       # notebooks для сравнительного анализа моделей
│   ├── custom_cnn/       # notebooks для моделей, обученных с нуля
│   ├── multiclass/       # notebooks для multi-class экспериментов
│   └── transfer_learning/ # notebooks для сравнения pretrained baseline-моделей
├── outputs/
│   ├── comparison/       # сравнение baseline и улучшенных экспериментов
│   ├── comparison_all/   # общее сравнение scratch и transfer learning моделей
│   ├── custom_cnn/       # результаты custom CNN from scratch
│   │   └── gradcam/      # Grad-CAM для custom CNN
│   ├── figures/          # графики, confusion matrix
│   └── gradcam/          # Grad-CAM тепловые карты
├── scripts/
│   ├── download_aptos.py
│   ├── check_dataset.py
│   └── train_baseline.py
├── src/
│   └── dr_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── explainability.py
│       └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Рекомендуемая версия Python

Для deep learning библиотек лучше использовать `Python 3.11`. На вашей машине сейчас доступен `Python 3.13.3`, но часть библиотек вроде `torch` и `opencv-python` иногда устанавливается стабильнее именно на `3.11`.

## Как положить датасет

### Вариант 1: скачать APTOS 2019 через KaggleHub

После установки зависимостей:

```bash
python3 scripts/download_aptos.py
```

Если нужна авторизация через интерактивный логин:

```bash
python3 scripts/download_aptos.py --login
```

Скрипт попытается скачать датасет `mariaherrerot/aptos2019` в папку `data/raw/aptos2019` и затем подскажет, где он нашёл CSV-файлы и папки с изображениями.

### Вариант 1: APTOS 2019 в текущем проекте

Положите файлы так:

```text
data/raw/aptos2019/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/
│   └── train_images/
│       ├── 000c1434d8d7.png
│       ├── 001639a390f0.png
│       └── ...
├── val_images/
│   └── val_images/
└── test_images/
    └── test_images/
```

Этот проект умеет автоматически читать колонки `id_code` и `diagnosis`, которые используются в `APTOS 2019`, и работает с уже готовыми `train/valid/test` split-файлами.

### Вариант 2: другой датасет

Если у вас другой датасет, самый простой формат CSV такой:

```text
image_id,label
img_001,0
img_002,1
```

или

```text
image_path,label
class0/img_001.png,0
class1/img_002.png,1
```

## Установка

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Если у вас пока нет `python3.11`, можно сначала просто сохранить структуру проекта, а окружение настроить позже.

## Быстрая проверка датасета

После того как вы загрузите изображения и CSV:

```bash
python3 scripts/check_dataset.py \
  --csv data/raw/aptos2019/train_1.csv \
  --images-dir data/raw/aptos2019/train_images/train_images \
  --binary
```

Скрипт проверит:

- сколько строк в CSV;
- сколько классов получилось;
- есть ли отсутствующие изображения;
- удаётся ли прочитать первый sample.

## Базовый запуск обучения

Отредактируйте путь в `configs/default.yaml`, затем:

```bash
python3 scripts/train_baseline.py --config configs/default.yaml
```

## Следующий baseline-этап для диплома

Чтобы baseline выглядел как полноценный research block, лучше сравнить не одну pretrained модель, а несколько стандартных архитектур при одинаковых условиях.
Для этого я подготовил еще два конфига:

```bash
python3 scripts/train_baseline.py --config configs/densenet121_baseline.yaml
python3 scripts/train_baseline.py --config configs/efficientnet_b0_baseline.yaml
```

Идея этого этапа:

- `ResNet18` - компактный и понятный baseline;
- `DenseNet121` - популярная медицинская архитектура с сильным feature reuse;
- `EfficientNet-B0` - более современный и эффективный baseline.

Такой блок сравнения уже выглядит намного ближе к формату дипломной экспериментальной главы.

## Первый multi-class эксперимент

Для перехода от binary screening к исходной задаче градации тяжести DR я добавил отдельный конфиг:

```bash
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass.yaml
```

Что важно в этом этапе:

- используется тот же текущий `APTOS 2019` dataset, заново скачивать исходные данные не нужно;
- `binary_classification: false`, поэтому сохраняются исходные классы `0, 1, 2, 3, 4`;
- в loss добавлены `balanced class weights`, чтобы учесть дисбаланс классов;
- основная validation-метрика для выбора лучшей эпохи: `f1_macro`.

Такой multi-class запуск уже ближе к реальной clinical grading задаче, чем бинарный screening baseline.

## Пошаговые preprocessing-эксперименты для multi-class

Чтобы multi-class ветка выглядела как аккуратное исследование, а не как набор случайных настроек, я подготовил два следующих конфига:

```bash
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass_crop.yaml
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass_crop_clahe.yaml
```

Логика этой последовательности:

- `efficientnet_b0_multiclass.yaml` - baseline без preprocessing;
- `efficientnet_b0_multiclass_crop.yaml` - только retinal-focused crop;
- `efficientnet_b0_multiclass_crop_clahe.yaml` - crop + CLAHE.

Так мы сможем честно проверить:

- дает ли пользy одно только центрирование retinal field;
- улучшает ли `CLAHE` различение близких стадий DR;
- какой preprocessing реально помогает редким severe-классам.

Для анализа результатов первого multi-class эксперимента я добавил notebook:

```text
notebooks/multiclass/11_efficientnet_b0_multiclass_results.ipynb
```

Он сохраняет:

- `Table 11.1` - основные multi-class метрики;
- `Table 11.2` - per-class precision, recall, F1-score, support;
- `Table 11.3` - confusion-matrix counts `5 x 5`;
- `Figure 11.1` - confusion matrix;
- `Figure 11.2` - row-normalized confusion matrix;
- `Figure 11.3` - per-class metric comparison.

Для сравнения multi-class baseline и preprocessing-вариантов я добавил еще один notebook:

```text
notebooks/multiclass/12_multiclass_preprocessing_comparison.ipynb
```

Он сохраняет:

- `Table 12.1` - comparison of overall multi-class metrics;
- `Table 12.2` - per-class F1-score comparison;
- `Table 12.3` - rare-class recall comparison for severe stages;
- `Figure 12.1` - overall metric comparison;
- `Figure 12.2` - per-class F1-score comparison;
- `Figure 12.3` - rare-class recall comparison.

## Следующий эксперимент после Grad-CAM

После explainability-анализа у нас появилась гипотеза, что модель частично использует края изображения, тени и область диска как shortcut-сигналы.
Для следующего более сильного эксперимента я добавил отдельный конфиг с предобработкой сетчатки:

- обрезка темных границ;
- приведение к квадратной области;
- CLAHE-контрастирование.

Запуск:

```bash
python3 scripts/train_baseline.py --config configs/resnet18_preprocessed.yaml
```

Так baseline в `configs/default.yaml` остается нетронутым, и мы можем сделать честное сравнение `baseline vs preprocessing-enhanced`.

## Эксперимент с собственной CNN from scratch

Для сравнения transfer learning с обучением с нуля я добавил отдельный конфиг:

```bash
python3 scripts/train_baseline.py --config configs/custom_cnn_scratch.yaml
```

Особенности этого эксперимента:

- `model_name: custom_cnn`
- `pretrained: false`
- отдельная папка для весов: `models/custom_cnn_scratch`
- отдельная папка для результатов: `outputs/custom_cnn`

## Ноутбук сравнения моделей

Для сравнения исходного baseline и preprocessing-enhanced версии я добавил ноутбук:

```text
notebooks/04_model_comparison.ipynb
```

Он сохраняет:

- `Table 4.1` - сравнение основных метрик;
- `Table 4.2` - сравнение `TP/TN/FP/FN`;
- `Table 4.3` - summary изменений предсказаний;
- `Figure 4.1` - сравнение метрик;
- `Figure 4.2` - trade-off между `FP` и `FN`;
- `Figure 4.3` - сравнение ROC-кривых.

## Ноутбук общего сравнения всех моделей

Для общего сравнения `custom_cnn`, `baseline ResNet18`, `preprocessed ResNet18`, `DenseNet121` и `EfficientNet-B0` я добавил отдельный notebook:

```text
notebooks/comparison/06_all_model_comparison.ipynb
```

Он сохраняет:

- `Table 6.1` - сравнение основных метрик всех baseline-моделей;
- `Table 6.2` - сравнение `TP/TN/FP/FN` всех baseline-моделей;
- `Table 6.3` - clinical summary с акцентом на `recall` и `false negatives`;
- `Figure 6.1` - сравнение test metrics;
- `Figure 6.2` - comparison of false positives and false negatives;
- `Figure 6.3` - comparison of ROC curves.

## Ноутбуки для transfer learning baseline-моделей

Для отдельных result-analysis notebooks по pretrained моделям я добавил:

```text
notebooks/transfer_learning/08_densenet121_results.ipynb
notebooks/transfer_learning/09_efficientnet_b0_results.ipynb
```

Оба ноутбука сохраняют:

- итоговую таблицу метрик на test split;
- таблицу с `TP/TN/FP/FN`;
- `confusion matrix`;
- `ROC curve`.

Для explainability-анализа сильнейшей transfer learning модели я также добавил:

```text
notebooks/transfer_learning/10_efficientnet_b0_gradcam.ipynb
```

Он сохраняет:

- `Table 10.1` - summary по `TP/TN/FP/FN` outcome categories;
- `Figure 10.1` - распределение outcome categories;
- `Image 10.1` - true positive Grad-CAM examples;
- `Image 10.2` - true negative Grad-CAM examples;
- `Image 10.3` - false positive Grad-CAM examples;
- `Image 10.4` - false negative Grad-CAM examples.

Результаты сохраняются в:

```text
outputs/efficientnet_b0_baseline/gradcam/
```

## Ноутбук для custom CNN

Для модели, обученной с нуля, я сохранил отдельный notebook в собственной папке:

```text
notebooks/custom_cnn/05_custom_cnn_results.ipynb
```

Это помогает не смешивать:

- `ResNet18` experiments;
- preprocessing experiments;
- `custom CNN from scratch`.

Для explainability-анализа `custom CNN` я добавил отдельный notebook:

```text
notebooks/custom_cnn/07_custom_cnn_gradcam.ipynb
```

Он сохраняет:

- `Table 7.1` - summary по `TP/TN/FP/FN` для custom CNN;
- `Figure 7.1` - распределение outcome categories;
- `Image 7.1` - true positive Grad-CAM examples;
- `Image 7.2` - true negative Grad-CAM examples;
- `Image 7.3` - false positive Grad-CAM examples;
- `Image 7.4` - false negative Grad-CAM examples.

## Что делать дальше

1. Загрузить один датасет, лучше начать с `APTOS 2019`.
2. Если скачивали через `KaggleHub`, при необходимости сверить реальные пути к `train_1.csv`, `valid.csv`, `test.csv` и папкам с изображениями.
3. Проверить чтение через `scripts/check_dataset.py`.
4. Запустить baseline-модель `ResNet18`.
5. Построить confusion matrix, ROC curve и анализ ошибок.
6. Выполнить Grad-CAM-анализ.
7. Запустить улучшенный `ResNet18` с предобработкой и сравнить результаты.
# Explainable-Deep-Learning-for-Detection-of-Diabetic-Retinopathy-from-Retinal-Fundus-Images.
