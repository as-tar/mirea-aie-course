# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано

- Выбран датасет EMNIST.
- Часть A: Dropout, BatchNorm, EarlyStopping. Часть B: большой LR, маленький LR, SGD с momentum и weight decay.

## 2. Среда и воспроизводимость

- Python: 3.12
- torch / torchvision: 2.10.0+cu128 / 0.25.0+cu128
- Устройство (CPU/GPU): GPU
- Seed: 42
- Как запустить: открыть `HW08-09.ipynb` и выполнить Run All.

## 3. Данные

- Датасет: EMNIST Balanced
- Разделение: train split 90/10 + test из torchvision
- Трансформации (transform): ToTensor, Normalize
- Комментарий: 47 классов, размерность 28x28, изображения в оттенках серого.

## 4. Базовая модель и обучение

- Модель MLP: 2 скрытых слоя 256 и 128, активация ReLU
- Loss: CrossEntropyLoss
- Базовый Optimizer (для части A): Adam (lr=1e-3)
- Batch size: 256
- Epochs (макс): 6, кроме E4 (10) и O3 (12)
- EarlyStopping: (patience=3, metric=val_accuracy)

## 5. Часть A (S08): регуляризация (E1-E4)

- E1 (base): 2 скрытых слоя, без Dropout/BatchNorm
- E2 (Dropout): как E1 + Dropout(p=0.2)
- E3 (BatchNorm): как E1 + BatchNorm
- E4 (EarlyStopping): лучший из (E2/E3) + EarlyStopping

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- O1: LR слишком большой (Adam, lr=1e-1)
- O2: LR слишком маленький (Adam, lr=1e-5)
- O3: SGD+momentum (momentum=0.9) + weight_decay=1e-4 (lr=1e-3)

## 7. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_model.pt`
- Конфиг лучшей модели: `./artifacts/best_config.json`
- Кривые лучшего прогона: `./artifacts/figures/curves_best.png`
- Кривые “плохих LR”: `./artifacts/figures/curves_lr_extremes.png`

Короткая сводка:

- Лучший эксперимент части A: E4
- Лучшая val_accuracy: 0.852
- Итоговая test_accuracy (для лучшей модели): 0.843
- Что видно на O1 (слишком большой LR): в данном случае особо плохого поведения замечено не было.
- Что видно на O2 (слишком маленький LR): модель начинает обучение с низкой accuracy и высокого loss и не успевает за то же количество эпох достичь уровня предыдущих моделей.
- Как повёл себя O3 (SGD+momentum + weight decay) относительно Adam (по кривым/метрике): обучение более плавное, но более медленное.

## 8. Анализ

- Dropout не особо помог, а вот BatchNorm улучшил accuracy.

## 9. Итоговый вывод

- Какой конфиг вы бы взяли как базовый и почему: E4, показал лучший результат.
- Что бы попробовали улучшить дальше: поэкспериментировать с архитектурой сети.

## 10. Приложение (опционально)

Приложения нет.
