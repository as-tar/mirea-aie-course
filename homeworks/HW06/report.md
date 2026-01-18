# HW06 – Report

## 1. Dataset

- Датасет: `S06-hw-dataset-0X.csv`
- Размер: (25000, 62)
- Целевая переменная: `target` (класс `0` - 95%, класс `1` - 5%)
- Признаки: числовые

## 2. Protocol

- Разбиение: train/test (в соотношении 80%/20%, `random_state`=42)
- Подбор: CV на train (5 фолдов)
- Метрики: accuracy, F1, ROC-AUC (т.к. бинарная классификация)

## 3. Models

- DummyClassifier (`strategy`="most_frequent")
- LogisticRegression (подбор гиперпараметров: `penalty`, `C`)
- DecisionTreeClassifier (подбор гиперпараметров: `max_depth`, `min_samples_leaf`, `ccp_alpha`)
- RandomForestClassifier (подбор гиперпараметров: `max_depth`, `min_samples_leaf`, `max_features`)
- AdaBoost (подбор гиперпараметров: `n_estimators`, `learning_rate`)

## 4. Results

- Dummy Classifier:
    - Accuracy: 0.9508
    - F1 Score: 0.0
    - ROC-AUC: 0.5
- Logistic Regression:
    - Accuracy: 0.9592
    - F1 Score: 0.3013698630136986
    - ROC-AUC: 0.8343047019027194
- Decision Tree:
    - Accuracy: 0.9672
    - F1 Score: 0.59
    - ROC-AUC: 0.8010195094588723
- Random Forest:
    - Accuracy: 0.9684
    - F1 Score: 0.5297619047619048
    - ROC-AUC: 0.896127693923132
- AdaBoost:
    - Accuracy: 0.9632
    - F1 Score: 0.4491017964071856
    - ROC-AUC: 0.8682662610176797

- Победитель по метрике ROC-AUC: Random Forest

## 5. Analysis

- Ошибки: см. файл `artifacts/figures/confusion_matrix.png`
- Интерпретация: см. файл `artifacts/figures/permutation_importance.png`

## 6. Conclusion

Ансамбли деревьев решений превосходят результаты отдельных деревьев.
