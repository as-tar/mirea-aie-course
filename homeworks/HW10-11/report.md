# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет `Flowers102`.
- Для части B выбран трек `detection` и датасет `Pascal VOC`.

## 2. Среда и воспроизводимость

- Python: 3.12
- torch / torchvision: 2.10.0+cu128 / 0.25.0+cu128
- Устройство (CPU/GPU): GPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `Flowers102`
- Разделение:
    - Train: 1020
    - Val: 1020
    - Test: 6149
- Базовые transforms: Resize, ToTensor, Normalize
- Augmentation transforms: RandomHorizontalFlip, RandomRotation
- Комментарий: 102 categories, each class consists of between 40 and 258 images.

### 3.2. Часть B: structured vision

- Датасет: `Pascal VOC` (`VOCDetection`)
- Трек: `detection`
- Что считается ground truth: bounding boxes класса `person`.

## 4. Часть A: модели и обучение (C1-C4)

- C1 (simple-cnn-base): простая CNN без аугментаций.
- C2 (simple-cnn-aug): та же CNN, но с разумными аугментациями.
- C3 (resnet18-head-only): `ResNet18` с pretrained weights; backbone заморожен, обучается только классификационная голова.
- C4 (resnet18-finetune): `ResNet18` с pretrained weights; частичное fine-tune (`layer4 + fc`).

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam
- Batch size: 128
- Epochs (макс): 10
- Критерий выбора лучшей модели: best val accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- Модель: `FasterRCNN_ResNet50_FPN_V2`
- V1: `score_threshold = 0.3`
- V2: `score_threshold = 0.7`

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples.png`, `./artifacts/figures/detection_metrics.png`

- Лучший эксперимент части A: C4
- Лучшая `val_accuracy`: 0.897
- Итоговая `test_accuracy` лучшего классификатора: 0.876
- Что дали аугментации (C2 vs C1): прирост метрики accuracy на 10%
- Что дал transfer learning (C3/C4 vs C1/C2): улучшение качества почти в 3 раза
- Что оказалось лучше: head-only или partial fine-tuning: partial fine-tuning
- Что показал режим V1 во второй части: считались только предсказания с уверенностью модели больше 0.3
- Что показал режим V2 во второй части: считались только предсказания с уверенностью модели больше 0.7
- Как интерпретируются метрики второй части:
    - Precision: доля истинных обнаружений среди всех объектов, которые модель пометила как "человек".
    - Recall: доля найденных моделью людей среди всех реально находящихся (как минимум размеченных) на изображении.
    - F1: среднее гармоническое между precision и recall.
    - IoU (Intersection over Union): мера того, насколько точно предсказанная рамка совпадает с истинной.

## 7. Анализ

- На датасете `Flowers102` с 1020 обучающими изображениями простая CNN показывает явную тенденцию к переобучению.
- Аугментации дали устойчивое улучшение.
- Pretrained ResNet18 показала себя лучше простой CNN, потому что уже обладала некоторым пониманием изображений, что особенно помогло при относительно малом наборе обучающих данных.
- Partial fine-tuning показал лучшие результаты, чем head-only.
- При переходе от V1 к V2 метрика precision растёт, а метрика recall падает.

## 8. Итоговый вывод

- Transfer Learning позволяет значительно улучшить качество в данном случае классификации изображений.
- В задаче detection важно грамотно подбирать уровень уверенности модели, балансируя между precision и recall.

## 9. Приложение (опционально)

Приложения нет.