"""Task 5 model training and MLflow tracking workflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COLUMN = "is_high_risk"
EXPERIMENT_NAME = "credit-risk-proxy-training"


def load_processed_data(path: str) -> pd.DataFrame:
    """Load processed customer-level dataset."""
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix and target vector."""
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    drop_cols = [target_column]
    if "CustomerId" in df.columns:
        drop_cols.append("CustomerId")

    X = df.drop(columns=drop_cols)
    y = df[target_column].astype(int)
    return X, y


def evaluate_binary_classifier(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute core binary classification metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def train_and_track(
    data_path: str = "data/processed/model_features.csv",
    experiment_name: str = EXPERIMENT_NAME,
    registered_model_name: str = "CreditRiskProxyModel",
) -> dict[str, float]:
    """Train candidate models, log to MLflow, and persist best model."""
    df = load_processed_data(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=2000,
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 50.0],
                "solver": ["liblinear", "lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 8, 12, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
    }

    mlflow.set_experiment(experiment_name)

    best_model = None
    best_name = ""
    best_metrics: dict[str, float] = {}
    best_roc_auc = -1.0

    for model_name, (base_model, param_space) in candidates.items():
        with mlflow.start_run(run_name=model_name):
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=12,
                scoring="roc_auc",
                cv=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)

            tuned_model = search.best_estimator_
            metrics = evaluate_binary_classifier(tuned_model, X_test, y_test)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_best_roc_auc", search.best_score_)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(
                sk_model=tuned_model,
                artifact_path="model",
            )

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model = tuned_model
                best_name = model_name
                best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    Path("models").mkdir(parents=True, exist_ok=True)
    local_model_path = Path("models") / "best_model.joblib"
    joblib.dump(best_model, local_model_path)

    with mlflow.start_run(run_name="best_model_registration"):
        mlflow.log_param("best_model_name", best_name)
        for metric_name, metric_value in best_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
        )

        # Local tracking stores may not support model registry, so keep safe.
        try:
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_model_name,
            )
        except Exception:
            pass

    return best_metrics


def main() -> None:
    """CLI entrypoint for Task 5 training."""
    metrics = train_and_track()
    print("Best model metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
