import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils import class_weight
from collections import Counter
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

class IljicevsModel:
    def __init__(self, models=None, param_grids=None):
        """
        Инициализация ансамбля моделей.
        :param models: Словарь с моделями {'имя модели': модель}. Можно не задавать для AutoML.
        :param param_grids: Словарь с гиперпараметрами {'имя модели': параметры}.
        """
        self.models = models
        self.param_grids = param_grids
        self.best_models = None
        self.selected_models = None

    def fit(self, X_train, y_train):
        """
        Обучение выбранных моделей.
        :param X_train: Признаки для обучения.
        :param y_train: Метки для обучения.
        """
        for model in self.selected_models:
            model.fit(X_train, y_train)
            
    def check_class_balance(self, y):
        """
        Проверка баланса классов и предложение решения при несбалансированных данных.
        :param y: Метки классов.
        """
        class_counts = Counter(y)
        total_samples = len(y)
        for cls, count in class_counts.items():
            print(f"Класс {cls}: {count} примеров ({count / total_samples:.2%})")
        
        if min(class_counts.values()) / max(class_counts.values()) < 0.5:
            print("Предупреждение: классы несбалансированы!")
            print("Решения: можно использовать взвешивание классов или выборку данных.")

    def tune_hyperparameters(self, X_train, y_train):
        """
        Настройка гиперпараметров с помощью GridSearchCV для каждой модели.
        :param X_train: Признаки для обучения.
        :param y_train: Метки для обучения.
        """
        self.best_models = {}
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, self.param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.best_models[name] = grid_search.best_estimator_
            print(f"Наилучшие параметры для {name}: {grid_search.best_params_}")

    def select_best_models(self, X_train, y_train, top_n=2):
        """
        Выбор лучших моделей на основе кросс-валидации.
        :param X_train: Признаки для обучения.
        :param y_train: Метки для обучения.
        :param top_n: Количество лучших моделей для выбора.
        """
        mean_scores = {}
        for name, model in self.best_models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_scores[name] = np.mean(scores)

        sorted_models = sorted(mean_scores.items(), key=lambda item: item[1], reverse=True)
        self.selected_models = [self.best_models[name] for name, _ in sorted_models[:top_n]]
        print(f"Выбраны модели: {[name for name, _ in sorted_models[:top_n]]}")

    def feature_importance(self):
        """
        Вычисление и визуализация важности признаков для ансамбля моделей.
        """
        importances = np.zeros(self.selected_models[0].n_features_in_)
        for model in self.selected_models:
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_

        importances /= len(self.selected_models)

        plt.bar(range(len(importances)), importances)
        plt.xlabel("Признаки")
        plt.ylabel("Важность")
        plt.title("Важность признаков для ансамбля моделей")
        plt.show()

    def cross_validate_with_metrics(self, X_train, y_train, metrics=None):
        """
        Проведение кросс-валидации с вычислением настраиваемых метрик.
        :param X_train: Признаки для обучения.
        :param y_train: Метки для обучения.
        :param metrics: Метрики для оценки, например ['accuracy', 'f1', 'roc_auc'].
        """
        if metrics is None:
            metrics = ['accuracy']

        for model in self.selected_models:
            print(f"Модель: {model.__class__.__name__}")
            for metric in metrics:
                if metric == 'accuracy':
                    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    print(f"Точность: {np.mean(score):.4f}")
                elif metric == 'f1':
                    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
                    print(f"F1-score: {np.mean(score):.4f}")
                elif metric == 'roc_auc':
                    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                    print(f"ROC AUC: {np.mean(score):.4f}")
                # Можно добавить другие метрики по аналогии

    def weighted_average_predictions(self, X_test, y_test):
        """
        Усреднение предсказаний с учетом весов моделей.
        :param X_test: Признаки для тестирования.
        :param y_test: Метки для тестирования.
        :return: Финальные предсказания.
        """
        weights = []
        for model in self.selected_models:
            scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            weights.append(np.mean(scores))

        total_proba = np.zeros((X_test.shape[0], len(np.unique(y_test))))
        for i, model in enumerate(self.selected_models):
            proba = model.predict_proba(X_test)
            total_proba += weights[i] * proba

        avg_proba = total_proba / sum(weights)
        final_predictions = np.argmax(avg_proba, axis=1)
        return final_predictions

    def score(self, X_test, y_test):
        """
        Оценка точности модели.
        :param X_test: Признаки для тестирования.
        :param y_test: Метки для тестирования.
        :return: Точность.
        """
        predictions = self.weighted_average_predictions(X_test, y_test)
        return accuracy_score(y_test, predictions)
