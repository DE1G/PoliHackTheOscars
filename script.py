import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import os
import matplotlib.pyplot as plt

# Load CSV file
file_path = "data/train/forecast_full.csv"
df = pd.read_csv(file_path, parse_dates=["date"])


# Select only the required columns
cols = ["date", "category", "state", "age_group", "gender", "is_weekend", "is_holiday"]
df = df[cols + ["actual_volume"]]  # assuming 'actual_volume' is our prediction target

# Feature engineering: date components
df["day"] = df.date.dt.day
df["month"] = df.date.dt.month
df["year"] = df.date.dt.year
df["weekday"] = df.date.dt.weekday

# Encode categorical variables using LabelEncoder
cat_features = ["category", "state", "age_group", "gender"]
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
FEATURES = ["category", "state", "age_group", "gender", "day", "month", "year", "weekday"]
TARGET = "actual_volume"
X = df[FEATURES]
y = df[TARGET]


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# XGBoost params
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "eta": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

# Train model
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20,
    verbose_eval=10
)

# Evaluate
y_pred = bst.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mas = mean_absolute_error(y_val, y_pred)
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation MAE: {mas:.2f}")

# Save the model
model_path = os.path.join(os.getcwd(), "xgb_model_forecast.json")
bst.save_model(model_path)
print(f"Model saved to {model_path}")

# Load the model later when needed
# This would typically be placed in a prediction script or another pipeline
loaded_model = xgb.Booster()
loaded_model.load_model(model_path)
print("Model loaded successfully for future predictions.")

# Example of using the loaded model for prediction:
# predictions = loaded_model.predict(xgb.DMatrix(X_val))

# Plotting the difference between actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_val)), y_val.values, label='Actual', alpha=0.7)
plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Volume')
plt.xlabel('Sample Index')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def create_lag_features(df, group_cols, target_col, lags):
    dfa = df.sort_values(by=[*group_cols, "date"])
    for lag in lags:
        dfa[f"{target_col}_lag_{lag}"] = dfa.groupby(group_cols)[target_col].shift(lag)
    return dfa

def create_rolling_features(df, group_cols, target_col, windows):
    dfb = df.sort_values(by=[*group_cols, "date"])
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df.groupby(group_cols)[target_col].transform(lambda x: x.shift(1).rolling(window).mean())
    return df

def create_diff_features(df, group_cols, target_col, diffs):
    dfc = df.sort_values(by=[*group_cols, "date"])
    for d in diffs:
        df[f"{target_col}_diff_{d}"] = df.groupby(group_cols)[target_col].diff(d)
    return df


