import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils import class_weight
from collections import Counter
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE
import logging
import joblib
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer, precision_recall_curve
from tpot import TPOTClassifier

# Логирование
logging.basicConfig(filename='model_training.log', level=logging.INFO)

# Инкапсуляция и расширяемость
class ModelTuner:
    def __init__(self, model, param_grid=None, search_method="grid", max_evals=50):
        """
        Класс для настройки гиперпараметров моделей с использованием GridSearchCV или Hyperopt.
        
        :param model: Модель, которая будет настроена.
        :param param_grid: Словарь параметров для поиска гиперпараметров (например, для GridSearchCV).
        :param search_method: Строка, указывающая метод поиска гиперпараметров ('grid' или 'hyperopt').
        :param max_evals: Количество итераций для байесовской оптимизации (Hyperopt).
        """
        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.max_evals = max_evals

    def tune(self, X_train, y_train):
        """
        Метод для запуска процесса поиска гиперпараметров.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: Лучшая модель, подобранная с использованием выбранного метода.
        """
        if self.search_method == "grid":
            return self.grid_search(X_train, y_train)
        elif self.search_method == "hyperopt":
            return self.hyperopt_search(X_train, y_train)

    def grid_search(self, X_train, y_train):
        """
        Настройка гиперпараметров с использованием GridSearchCV.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: Лучшая модель по результатам GridSearchCV.
        """
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def hyperopt_search(self, X_train, y_train):
        """
        Настройка гиперпараметров с использованием Hyperopt (байесовская оптимизация).
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: Лучшая модель по результатам байесовской оптимизации.
        """
        def objective(params):
            model = self.model
            model.set_params(**params)
            score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=self.param_grid, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        self.model.set_params(**best)
        return self.model


class IljicevsModel:
    def __init__(self, models=None, param_grids=None, search_method="grid"):
        """
        Инициализация ансамбля моделей с возможностью выбора метода поиска гиперпараметров.
        
        :param models: Словарь с моделями, где ключ — название модели, а значение — объект модели.
        :param param_grids: Словарь сеток гиперпараметров для каждой модели.
        :param search_method: Строка, указывающая метод поиска гиперпараметров ('grid' или 'hyperopt').
        """
        self.models = models
        self.param_grids = param_grids
        self.best_models = None
        self.selected_models = None
        self.search_method = search_method

    def fit(self, X_train, y_train):
        """
        Обучение выбранных моделей на обучающем наборе данных.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        """
        for model in self.selected_models:
            model.fit(X_train, y_train)

    def check_class_balance(self, X_train, y_train):
        """
        Проверка баланса классов и балансировка данных с использованием SMOTE при необходимости.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: (X_res, y_res) - сбалансированные признаки и метки классов или оригинальные данные, если балансировка не требуется.
        """
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        for cls, count in class_counts.items():
            print(f"Класс {cls}: {count} примеров ({count / total_samples:.2%})")

        if min(class_counts.values()) / max(class_counts.values()) < 0.5:
            print("Предупреждение: классы несбалансированы!")
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X_train, y_train)
            print("Применен SMOTE для балансировки классов.")
            return X_res, y_res
        return X_train, y_train

    def tune_hyperparameters(self, X_train, y_train):
        """
        Настройка гиперпараметров с использованием ModelTuner.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        """
        self.best_models = {}
        for name, model in self.models.items():
            tuner = ModelTuner(model, self.param_grids[name], search_method=self.search_method)
            self.best_models[name] = tuner.tune(X_train, y_train)
            print(f"Наилучшие параметры для {name}")

    def select_best_models(self, X_train, y_train, top_n=2):
        """
        Выбор лучших моделей на основе кросс-валидации.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :param top_n: Количество лучших моделей для выбора в ансамбль.
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
        sns.barplot(x=np.arange(len(importances)), y=importances)
        plt.xlabel("Признаки")
        plt.ylabel("Важность")
        plt.title("Важность признаков для ансамбля моделей")
        plt.show()

    def cross_validate_with_custom_metrics(self, X_train, y_train, custom_metrics=None):
        """
        Проведение кросс-валидации с вычислением настраиваемых метрик.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :param custom_metrics: Словарь метрик для оценки, например {'accuracy': make_scorer(accuracy_score)}.
        """
        if custom_metrics is None:
            custom_metrics = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score, average='macro'),
                'roc_auc': make_scorer(roc_auc_score, multi_class='ovr')
            }

        for model in self.selected_models:
            print(f"Модель: {model.__class__.__name__}")
            for metric_name, scorer in custom_metrics.items():
                score = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
                print(f"{metric_name}: {np.mean(score):.4f}")

    def weighted_average_predictions(self, X_test, y_test):
        """
        Усреднение предсказаний с учетом весов моделей.
        
        :param X_test: Признаки тестового набора данных.
        :param y_test: Метки классов тестового набора данных.
        :return: Предсказанные метки классов.
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
        Оценка точности ансамбля моделей на тестовых данных.
        
        :param X_test: Признаки тестового набора данных.
        :param y_test: Метки классов тестового набора данных.
        :return: Точность предсказаний ансамбля моделей.
        """
        predictions = self.weighted_average_predictions(X_test, y_test)
        return accuracy_score(y_test, predictions)

    def plot_precision_recall_curve(self, X_test, y_test):
        """
        Визуализация Precision-Recall кривой для ансамбля моделей.
        
        :param X_test: Признаки тестового набора данных.
        :param y_test: Метки классов тестового набора данных.
        """
        for model in self.selected_models:
            proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, proba)
            plt.plot(recall, precision, label=f'{model.__class__.__name__}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def parallel_cross_val(self, X_train, y_train):
        """
        Параллельное выполнение кросс-валидации для всех выбранных моделей.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: Результаты кросс-валидации для каждой модели.
        """
        results = Parallel(n_jobs=-1)(delayed(cross_val_score)(model, X_train, y_train, cv=5) for model in self.selected_models)
        return results

    def stability_metric(self, X_test):
        """
        Оценка стабильности предсказаний ансамбля моделей.
        
        :param X_test: Признаки тестового набора данных.
        :return: Метрика стабильности ансамбля (чем ближе к 1, тем стабильнее предсказания).
        """
        predictions = [model.predict(X_test) for model in self.selected_models]
        std_dev = np.std(predictions, axis=0)
        stability_score = 1 - np.mean(std_dev)
        print(f"Стабильность ансамбля: {stability_score:.4f}")
        return stability_score

    def save_model(self, model, filename):
        """
        Сохранение модели на диск.
        
        :param model: Модель для сохранения.
        :param filename: Имя файла для сохранения модели.
        """
        joblib.dump(model, filename)
        print(f"Модель сохранена в {filename}")

    def load_model(self, filename):
        """
        Загрузка модели с диска.
        
        :param filename: Имя файла с моделью.
        :return: Загруженная модель.
        """
        model = joblib.load(filename)
        print(f"Модель загружена из {filename}")
        return model

    def run_automl(self, X_train, y_train):
        """
        Запуск AutoML с использованием TPOT для автоматического подбора моделей и гиперпараметров.
        
        :param X_train: Признаки обучающего набора данных.
        :param y_train: Метки классов обучающего набора данных.
        :return: Лучший пайплайн моделей, найденный TPOT.
        """
        automl = TPOTClassifier(generations=5, population_size=50, cv=5, verbosity=2, random_state=42)
        automl.fit(X_train, y_train)
        print("Лучший пайплайн TPOT:", automl.fitted_pipeline_)
        return automl.fitted_pipeline_
