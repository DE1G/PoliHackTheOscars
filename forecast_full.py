import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import pickle

# Global paths
MODEL_PATH = os.path.join(os.getcwd(), "xgb_model_forecast.json")
ENCODER_PATH = os.path.join(os.getcwd(), "label_encoders.pkl")

# Features used by the model
FEATURES = [
    "category", "state", "age_group", "gender", "expected_volume", "day", "month", "year", "weekday",
    "is_weekend", "is_holiday", "expected_volume_lag_1", "expected_volume_lag_7", "expected_volume_lag_14",
    "expected_volume_lag_28", "expected_volume_roll_mean_7", "expected_volume_roll_mean_14", "expected_volume_roll_mean_28"
]

# Global label encoders
label_encoders = {}

def preprocess(df: pd.DataFrame, fit_encoders: bool = True):
    global label_encoders
    df = df.copy()

    # Date features
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["year"] = df.date.dt.year
    df["weekday"] = df.date.dt.weekday

    # Encode categorical features
    cat_features = ["category", "state", "age_group", "gender", "is_weekend", "is_holiday"]
    for col in cat_features:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is None:
                raise ValueError(f"Label encoder for column '{col}' not found. Did you load them?")
            df[col] = le.transform(df[col])

    # Lag features
    for lag in [1, 7, 14, 28]:
        df[f'expected_volume_lag_{lag}'] = df['expected_volume'].shift(lag)

    # Rolling mean features
    df['expected_volume_roll_mean_7'] = df['expected_volume'].rolling(7).mean()
    df['expected_volume_roll_mean_14'] = df['expected_volume'].rolling(14).mean()
    df['expected_volume_roll_mean_28'] = df['expected_volume'].rolling(28).mean()

    # Drop rows with NaN (due to rolling/lag features)
    df = df.dropna()

    return df

def train_model(train_csv_path="data/train/forecast_full.csv"):
    global label_encoders

    df = pd.read_csv(train_csv_path, parse_dates=["date"])
    df = preprocess(df, fit_encoders=True)

    X = df[FEATURES]
    y = df["actual_volume"]
    dtrain = xgb.DMatrix(X, label=y)

    # XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train")],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    # Save model and encoders
    model.save_model(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoders, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoders saved to {ENCODER_PATH}")
    return model

def load_model():
    global label_encoders
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    print("Model and label encoders loaded successfully.")
    return model

def predict(model, df_val: pd.DataFrame):
    df_val = preprocess(df_val, fit_encoders=False)
    dval = xgb.DMatrix(df_val[FEATURES])
    preds = model.predict(dval)
    return df_val, preds

def evaluate(preds, y_true):
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"Validation RMSE: {rmse:.2f}")
    return rmse

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Volume"):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_true)), y_true, label='Actual', alpha=0.7)
    plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Optional: run training and evaluation when script is executed directly
if __name__ == "__main__":
    model = train_model()
    df_val = pd.read_csv("data/val/forecast_full.csv", parse_dates=["date"])
    df_val, preds = predict(model, df_val)
    evaluate(preds, df_val["actual_volume"])
    plot_predictions(df_val["actual_volume"].values, preds)
