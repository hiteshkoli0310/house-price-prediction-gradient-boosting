# Bengaluru House Price Prediction (Gradient Boosting + XGBoost)

[Open in Colab](https://colab.research.google.com/github/hiteshkoli0310/house-price-prediction-gradient-boosting/blob/main/house_price_prediction.ipynb)

Predict Bengaluru house prices using tree-based boosting models with a clean, end-to-end workflow in a single notebook. The project covers data cleaning, feature engineering, one-hot encoding, an 80/20 train-test split, baseline Gradient Boosting, grid-search tuning, optional XGBoost, evaluation plots, and a tiny inference example.

Typical performance on the provided dataset is around R² ≈ 0.67 for a baseline Gradient Boosting model and ≈ 0.70–0.71 after hyperparameter tuning (results can vary by random seed and environment).

## What’s inside

- Jupyter notebook: `house_price_prediction.ipynb`
- Dataset CSV: `Bengaluru_House_Data (1).csv` (from Kaggle)
- This README with setup, usage, and troubleshooting tips

## Dataset

- Source: Kaggle – “Bengaluru House Data”
- Place the CSV in the same folder as the notebook. The notebook expects the file path:
  - `Bengaluru_House_Data.csv`
  - If your file is named `Bengaluru_House_Data (1).csv` (as in this repo), either:
  - Rename it to `Bengaluru_House_Data.csv`, or
  - Change the `file_path` variable near the top of the notebook accordingly.

## Setup (Windows PowerShell)

You’ll need Python 3.9+.

Option A — quick start with a virtual environment and requirements:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Option B — install packages manually (if you don’t want a requirements file):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

You can run the notebook in VS Code (recommended) or Jupyter:

```
python -m pip install notebook
jupyter notebook
```

## How the notebook works

1. Import libraries: pandas, numpy, matplotlib, seaborn, scikit-learn (modeling + metrics), and optionally xgboost.
2. Load data: reads the Kaggle CSV into a DataFrame.
3. Cleaning and feature engineering:
   - Drops low-signal columns: `availability`, `society`.
   - Extracts `BHK` from the textual `size` field.
   - Converts `total_sqft` to numeric, handling ranges like `1200-1500` by taking the mean.
   - Drops rows with missing values after conversions.
4. Encoding and split:
   - One-hot encodes `area_type` and `location` (with `drop_first=True`).
   - Train/test split = 80/20 with `random_state=42`.
5. Baseline model (Gradient Boosting):
   - `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`
   - Evaluates MAE, RMSE, and R².
6. Hyperparameter tuning:
   - `GridSearchCV` over `n_estimators`, `learning_rate`, and `max_depth`.
   - Trains a new model with best params; re-evaluates metrics.
7. Visualization:
   - Actual vs Predicted scatter plot.
   - Error distribution histogram.
8. Inference example:
   - Creates a single-row, one-hot-aligned sample and predicts a price using the tuned model.
9. Optional: XGBoost
   - Baseline `XGBRegressor` + `RandomizedSearchCV` for a broader parameter sweep.

## Results (indicative)

- Baseline Gradient Boosting: R² ≈ 0.67
- After tuning: R² ≈ 0.70–0.71

These numbers are illustrative; they may vary with environment, preprocessing decisions, and train/test split randomness.

## Project structure

```
banglore_house_price_prediction/
├── Bengaluru_House_Data (1).csv       
├── house_price_prediction.ipynb        
└── README.md                          
```

## Tips & troubleshooting

- File not found: If you see a file path error, make sure the CSV name matches `file_path` in the notebook.
- One-hot mismatch: When doing custom inference, ensure your sample DataFrame has all columns from `X_train` (including the correct one-hot names).
- Memory/runtime: XGBoost tuning can be heavier; reduce `n_iter`/`cv` or simplify the parameter space if needed.

## Acknowledgements

- Dataset: Kaggle – Bengaluru House Data
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

## License

This project is for educational purposes. If you plan to use it commercially, please review dataset terms on Kaggle and ensure compliance with library licenses.
