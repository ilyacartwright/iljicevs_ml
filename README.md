# Iljicevs ML

Iljicevs ML is a Python library designed to simplify the process of selecting, training, and optimizing multiple machine learning models in an ensemble approach. It provides functionalities for dynamic model selection, hyperparameter tuning, feature importance, cross-validation with multiple metrics, and more.

## Features

- **Model Hyperparameter Tuning**: Automatically search for the best hyperparameters using GridSearchCV.
- **Dynamic Model Selection**: Selects the best performing models based on cross-validation results.
- **Weighted Average Predictions**: Combines predictions from multiple models by averaging their outputs based on accuracy weights.
- **Feature Importance Visualization**: Displays the importance of features across ensemble models.
- **Class Balance Checking**: Automatically checks class balance and suggests solutions for unbalanced datasets.
- **Cross-validation with Metrics**: Supports evaluation with multiple metrics (Accuracy, F1, ROC AUC, etc.).
- **AutoML Support**: Automatically selects models based on dataset characteristics.

## Installation

To install the package, simply use `pip` (once it is published to PyPI):

```bash
pip install iljicevs_ml
```
Or, if you want to install directly from the source:

```bash
git clone https://github.com/yourusername/iljicevs_ml.git
cd iljicevs_ml
python setup.py install
```

## Usage

### Basic Example

Here's an example of how you can use iljicevs_ml to select models, tune their hyperparameters, and evaluate their performance.


```Python
from iljicevs_ml import IljicevsModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models and hyperparameters
models = {
    'logistic_regression': LogisticRegression(max_iter=200),
    'random_forest': RandomForestClassifier(),
    'svm': SVC(probability=True),
    'gradient_boosting': GradientBoostingClassifier()
}

param_grids = {
    'logistic_regression': {'C': [0.1, 1, 10]},
    'random_forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
}

# Initialize and train ensemble model
iljicevs = IljicevsModel(models, param_grids)

# Check class balance
iljicevs.check_class_balance(y_train)

# Tune hyperparameters
iljicevs.tune_hyperparameters(X_train, y_train)

# Select the best performing models
iljicevs.select_best_models(X_train, y_train)

# Fit the selected models
iljicevs.fit(X_train, y_train)

# Evaluate the ensemble's accuracy on the test set
accuracy = iljicevs.score(X_test, y_test)
print(f"Ensemble accuracy: {accuracy}")

# Display feature importance
iljicevs.feature_importance()

# Perform cross-validation with multiple metrics
iljicevs.cross_validate_with_metrics(X_train, y_train, metrics=['accuracy', 'f1', 'roc_auc'])
```

### Class Balance Checking
To avoid issues with unbalanced datasets, you can use the built-in method check_class_balance() to get a summary of the class distribution and suggestions for handling imbalances:
```python
iljicevs.check_class_balance(y_train)
```

### Feature Importance
To visualize the importance of features across models, use:
```python
iljicevs.feature_importance()
```

### Cross-Validation with Metrics
For more detailed model evaluation, you can use cross-validation with custom metrics:
```python
iljicevs.cross_validate_with_metrics(X_train, y_train, metrics=['accuracy', 'f1', 'roc_auc'])
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you find bugs or want to suggest new features.
1. Fork the repository
2. Create your feature branch (git checkout -b feature/my-feature)
3. Commit your changes (git commit -m 'Add some feature')
4. Push to the branch (git push origin feature/my-feature)
5. Open a Pull Request

### License
This project is licensed under the MIT License - see the LICENSE file for details.
