import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def engineer_features(df):
    """
    Feature engineering financier
    """

    eps = 1e-6

    # Calcul résultat si absent
    if 'net_result' not in df.columns:
        df['net_result'] = df['class_7_total'] - df['class_6_total']

    # Ratios financiers
    df['ebitda_margin'] = (df['class_7_total'] - df['class_6_total']) / (df['class_7_total'] + eps)
    df['capex_intensity'] = df['class_2_balance'] / (df['class_7_total'] + eps)

    # Historique profit
    df['profit_lag1'] = df['net_result'].shift(1)
    df['profit_lag2'] = df['net_result'].shift(2)
    df['profit_lag3'] = df['net_result'].shift(3)

    # Croissance du profit
    df['profit_growth'] = df['net_result'].pct_change()

    # Target futur (prévision)
    df['target'] = df['net_result'].shift(-1)

    # Nettoyage
    df = df.ffill().bfill()

    return df


def train_model(X_train, y_train):

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print("Evaluation du modèle")
    print("--------------------")
    print(f"R2 Score : {r2:.3f}")
    print(f"MAE      : {mae:.3f}")

    return preds


def show_feature_importance(model, features):

    importance = model.feature_importances_

    fi = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print("\nFeature importance")
    print(fi)

    return fi


def run_predictive_engine(csv_path):

    # Chargement
    df = pd.read_csv(csv_path)

    # Tri chronologique si colonne date
    if 'year' in df.columns:
        df = df.sort_values('year')

    if 'date' in df.columns:
        df = df.sort_values('date')

    # Feature engineering
    df = engineer_features(df)

    features = [
        'class_2_balance',
        'class_6_total',
        'class_7_total',
        'inflation_index',
        'ecb_rate',
        'ebitda_margin',
        'capex_intensity',
        'profit_lag1',
        'profit_lag2',
        'profit_lag3',
        'profit_growth'
    ]

    # Vérification colonnes
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    X = df[features]
    y = df['target']

    # Suppression dernières lignes avec target NaN
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    # Split temporel
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Entraînement
    model = train_model(X_train, y_train)

    # Evaluation
    preds = evaluate_model(model, X_test, y_test)

    # Importance des variables
    show_feature_importance(model, features)

    return model


if __name__ == "__main__":

    model = run_predictive_engine("data/pme_financials_sample.csv")
