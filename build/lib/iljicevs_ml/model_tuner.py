from sklearn.model_selection import GridSearchCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

class ModelTuner:
    def __init__(self, model, param_grid=None, search_method="grid", max_evals=50):
        """
        Class for tuning model hyperparameters using GridSearchCV or Hyperopt (EN).
        Класс для настройки гиперпараметров моделей с использованием GridSearchCV или Hyperopt (RU).
        Klasse zur Abstimmung von Modellhyperparametern mit GridSearchCV oder Hyperopt (DE).
        使用 GridSearchCV 或 Hyperopt 调整模型超参数的类 (ZH).

        Parameters (EN):
        Параметры (RU):
        Parameter (DE):
        参数 (ZH):
        - model (object): The machine learning model to be tuned (EN).
                          Модель машинного обучения для настройки (RU).
                          Das zu optimierende Modell des maschinellen Lernens (DE).
                          要调整的机器学习模型 (ZH).
        - param_grid (dict or list of dicts): Hyperparameter grid for tuning (EN).
                                              Сетка гиперпараметров для настройки (RU).
                                              Hyperparameter-Raster zur Abstimmung (DE).
                                              调整的超参数网格 (ZH).
        - search_method (str): Search method ('grid' for GridSearchCV or 'hyperopt' for Hyperopt) (EN).
                               Метод поиска ('grid' для GridSearchCV или 'hyperopt' для Hyperopt) (RU).
                               Suchmethode ('grid' für GridSearchCV oder 'hyperopt' für Hyperopt) (DE).
                               搜索方法（'grid' 用于 GridSearchCV 或 'hyperopt' 用于 Hyperopt）(ZH).
        - max_evals (int): Maximum number of evaluations for Hyperopt (default: 50) (EN).
                           Максимальное количество итераций для Hyperopt (по умолчанию: 50) (RU).
                           Maximale Anzahl von Auswertungen für Hyperopt (Standard: 50) (DE).
                           Hyperopt 的最大评估次数 (默认: 50) (ZH).
        """
        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.max_evals = max_evals

    def tune(self, X_train, y_train):
        """
        Method to start the hyperparameter search process (EN).
        Метод для запуска процесса поиска гиперпараметров (RU).
        Methode zum Starten des Hyperparametersuchprozesses (DE).
        启动超参数搜索过程的方法 (ZH).

        Parameters (EN):
        Параметры (RU):
        Parameter (DE):
        参数 (ZH):
        - X_train (array-like): Training feature matrix (EN).
                                Матрица признаков для обучения (RU).
                                Merkmalsmatrix zum Training (DE).
                                用于训练的特征矩阵 (ZH).
        - y_train (array-like): Training target variable (EN).
                                Целевая переменная для обучения (RU).
                                Zielvariable zum Training (DE).
                                用于训练的目标变量 (ZH).

        Returns (EN):
        Возвращает (RU):
        Rückgabe (DE):
        返回 (ZH):
        - best_estimator (object): The best model after hyperparameter tuning (EN).
                                   Лучшая модель после настройки гиперпараметров (RU).
                                   Das beste Modell nach der Hyperparameter-Abstimmung (DE).
                                   超参数调整后的最佳模型 (ZH).
        """
        if self.search_method == "grid":
            return self.grid_search(X_train, y_train)
        elif self.search_method == "hyperopt":
            return self.hyperopt_search(X_train, y_train)

    def grid_search(self, X_train, y_train):
        """
        Hyperparameter tuning using GridSearchCV (EN).
        Настройка гиперпараметров с использованием GridSearchCV (RU).
        Hyperparameter-Abstimmung mit GridSearchCV (DE).
        使用 GridSearchCV 进行超参数调整 (ZH).

        Parameters (EN):
        Параметры (RU):
        Parameter (DE):
        参数 (ZH):
        - X_train (array-like): Training feature matrix (EN).
                                Матрица признаков для обучения (RU).
                                Merkmalsmatrix zum Training (DE).
                                用于训练的特征矩阵 (ZH).
        - y_train (array-like): Training target variable (EN).
                                Целевая переменная для обучения (RU).
                                Zielvariable zum Training (DE).
                                用于训练的目标变量 (ZH).

        Returns (EN):
        Возвращает (RU):
        Rückgabe (DE):
        返回 (ZH):
        - best_estimator (object): The best model after GridSearchCV tuning (EN).
                                   Лучшая модель после настройки с помощью GridSearchCV (RU).
                                   Das beste Modell nach der Abstimmung mit GridSearchCV (DE).
                                   使用 GridSearchCV 调整后的最佳模型 (ZH).
        """
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def hyperopt_search(self, X_train, y_train):
        """
        Hyperparameter tuning using Hyperopt (Bayesian optimization) (EN).
        Настройка гиперпараметров с использованием Hyperopt (байесовская оптимизация) (RU).
        Hyperparameter-Abstimmung mit Hyperopt (Bayes'sche Optimierung) (DE).
        使用 Hyperopt 进行超参数调整（贝叶斯优化）(ZH).

        Parameters (EN):
        Параметры (RU):
        Parameter (DE):
        参数 (ZH):
        - X_train (array-like): Training feature matrix (EN).
                                Матрица признаков для обучения (RU).
                                Merkmalsmatrix zum Training (DE).
                                用于训练的特征矩阵 (ZH).
        - y_train (array-like): Training target variable (EN).
                                Целевая переменная для обучения (RU).
                                Zielvariable zum Training (DE).
                                用于训练的目标变量 (ZH).

        Returns (EN):
        Возвращает (RU):
        Rückgabe (DE):
        返回 (ZH):
        - best_model (object): The best model after Hyperopt tuning (EN).
                               Лучшая модель после настройки с помощью Hyperopt (RU).
                               Das beste Modell nach der Abstimmung mit Hyperopt (DE).
                               使用 Hyperopt 调整后的最佳模型 (ZH).
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
