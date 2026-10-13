# California Housing: Random Forest vs. XGBoost

A machine learning project that compares Random Forest and XGBoost regressors on the California Housing dataset to predict median house values, and identifies which model performs best on this data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Methodology](#methodology)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

The goal of this project is to answer one question: for predicting California house prices, which of two popular tree ensemble methods gives the better result?

- **Random Forest** builds many decision trees on random bootstrap samples and averages their predictions. Trees are trained independently, which makes the method robust and hard to overfit with default settings.
- **XGBoost** (Extreme Gradient Boosting) builds trees sequentially, with each new tree correcting the errors of the previous ones. It usually needs more tuning but can reach higher accuracy.

Both models are trained on the same data splits, tuned with the same search budget, and evaluated with the same metrics so the comparison is fair.

## Dataset

The project uses the **California Housing** dataset, originally derived from the 1990 U.S. Census.

| Property | Value |
|---|---|
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples | 20,640 |
| Features | 8 numeric features |
| Target | `MedHouseVal`: median house value for a block group, in units of $100,000 |

**Features:**

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median age of houses in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

**Requirements:**

- Python 3.9+
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## Usage

Run the full pipeline from the command line:

```bash
python src/train.py      # Trains both models and saves them to models/
python src/evaluate.py   # Computes metrics and writes results/metrics.csv
```

Or open the notebook for a step-by-step walkthrough:

```bash
jupyter notebook notebooks/comparison.ipynb
```

## Methodology

1. **Data loading.** The dataset is fetched with scikit-learn and cached locally.
2. **Exploratory analysis.** Distributions, correlations, and outliers are inspected. Note that the target is capped at $500,000 in the original data.
3. **Train/test split.** 80% of the data is used for training and 20% for testing, with a fixed random seed for reproducibility.
4. **Model training.** Both models are tuned with `RandomizedSearchCV` and 5-fold cross-validation on the training set only, so the test set stays untouched until final evaluation.
5. **Evaluation.** The best configuration of each model is scored on the held-out test set.

**Hyperparameter search spaces:**

| Model | Parameters tuned |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |

**Metrics:**

- **RMSE** (Root Mean Squared Error): typical prediction error, in the same units as the target. Lower is better.
- **MAE** (Mean Absolute Error): average absolute error, less sensitive to large outliers. Lower is better.
- **R²** (Coefficient of Determination): share of variance explained. Higher is better.
- **Training time**: wall-clock time to fit the tuned model.

  ## Future Work

- Try LightGBM or CatBoost as additional gradient boosting baselines
- Add geographic features, such as distance to the coast or to major cities
- Use SHAP values to explain individual predictions
- Test on other housing datasets to check generalization
## Conclusion
Random Forest is the Winner
R² Score 0.8061
RMSE $50,410
Higher accuracy in price prediction
Lower prediction error vs XGBoost
Best performing model for California house price prediction

Key Findings
- Income is the #1 driver of house prices
- Location (coastal proximity) is critical
- House age and rooms also influence prices
- Random Forest has a strong baseline model with easier implementation and less tuning
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Dataset: Pace, R. Kelley and Ronald Barry (1997), "Sparse Spatial Autoregressions," *Statistics & Probability Letters*.
- Libraries: [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
