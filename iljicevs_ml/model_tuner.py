from sklearn.model_selection import GridSearchCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score

class ModelTuner:
    def __init__(self, model, param_grid=None, search_method="grid", max_evals=50):
        """
        Класс для настройки гиперпараметров моделей с использованием GridSearchCV или Hyperopt.
        """
        self.model = model
        self.param_grid = param_grid
        self.search_method = search_method
        self.max_evals = max_evals

    def tune(self, X_train, y_train):
        """
        Метод для запуска процесса поиска гиперпараметров.
        """
        if self.search_method == "grid":
            return self.grid_search(X_train, y_train)
        elif self.search_method == "hyperopt":
            return self.hyperopt_search(X_train, y_train)

    def grid_search(self, X_train, y_train):
        """
        Настройка гиперпараметров с использованием GridSearchCV.
        """
        grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def hyperopt_search(self, X_train, y_train):
        """
        Настройка гиперпараметров с использованием Hyperopt (байесовская оптимизация).
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
