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
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Create synthetic dataset for classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42, class_sep=1.5)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models and parameters for hyperparameter tuning
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

param_grids = {
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
}

# Create an instance of IljicevsModel with the selected models and hyperparameter grids
iljicevs_model = IljicevsModel(models=models, param_grids=param_grids, search_method="grid")

# Check and balance class distribution using SMOTE (if necessary)
X_train_bal, y_train_bal = iljicevs_model.check_class_balance(X_train, y_train)

# Tune hyperparameters using GridSearchCV
iljicevs_model.tune_hyperparameters(X_train_bal, y_train_bal)

# Select the top two models based on cross-validation results
iljicevs_model.select_best_models(X_train_bal, y_train_bal, top_n=2)

# Train the selected models
iljicevs_model.fit(X_train_bal, y_train_bal)

# Evaluate accuracy on the test set
accuracy = iljicevs_model.score(X_test, y_test)
print(f"Test set accuracy: {accuracy:.4f}")

# Visualize feature importance for the ensemble models
iljicevs_model.feature_importance()

# Plot Precision-Recall curve
iljicevs_model.plot_precision_recall_curve(X_test, y_test)

# Compute stability of the ensemble models
stability = iljicevs_model.stability_metric(X_test)
print(f"Ensemble stability: {stability:.4f}")

# Save and load the model
iljicevs_model.save_model(iljicevs_model.selected_models[0], 'best_model.pkl')
loaded_model = iljicevs_model.load_model('best_model.pkl')

# Evaluate the loaded model on the test set
loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
print(f"Loaded model accuracy: {loaded_accuracy:.4f}")

# Example of using AutoML via TPOT (optional)
# automl_pipeline = iljicevs_model.run_automl(X_train_bal, y_train_bal)
```

### Class Balance Checking
To avoid issues with unbalanced datasets, you can use the built-in method check_class_balance() to get a summary of the class distribution and suggestions for handling imbalances:
```python
iljicevs.check_class_balance(X_train, y_train)
```

### Feature Importance
To visualize the importance of features across models, use:
```python
iljicevs.feature_importance()
```

### Cross-Validation with Metrics
For more detailed model evaluation, you can use cross-validation with custom metrics:
```python
iljicevs.cross_validate_with_custom_metrics(X_train, y_train, custom_metrics=['accuracy', 'f1', 'roc_auc'])
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
