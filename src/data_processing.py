"""Data processing pipeline for credit risk modeling.

Task 3 deliverable: transform raw transaction-level data into a model-ready,
customer-level feature matrix using a single fitted sklearn Pipeline object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {
    "CustomerId",
    "AccountId",
    "SubscriptionId",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "CurrencyCode",
}

NUMERIC_FEATURES = [
    "total_txn_amount",
    "avg_txn_amount",
    "txn_count",
    "txn_amount_std",
    "total_value",
    "avg_value",
    "value_std",
    "account_count",
    "subscription_count",
    "provider_count",
    "product_count",
    "product_category_count",
    "channel_count",
    "country_code_count",
    "avg_txn_hour",
    "avg_txn_day",
    "avg_txn_month",
    "avg_txn_year",
    "avg_txn_dayofweek",
    "weekend_txn_ratio",
    "recency_days",
]

CATEGORICAL_FEATURES = [
    "primary_currency",
    "primary_channel",
    "primary_product_category",
    "primary_pricing_strategy",
]


def _safe_mode(series: pd.Series) -> str:
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return "unknown"
    return str(mode_values.iloc[0])


class CustomerFeatureBuilder(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level records into customer-level features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CustomerFeatureBuilder":
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(X)
        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], errors="coerce", utc=True
        )
        df = df.dropna(subset=["CustomerId", "TransactionStartTime"])

        # Time-based components from transaction timestamp.
        df["txn_hour"] = df["TransactionStartTime"].dt.hour
        df["txn_day"] = df["TransactionStartTime"].dt.day
        df["txn_month"] = df["TransactionStartTime"].dt.month
        df["txn_year"] = df["TransactionStartTime"].dt.year
        df["txn_dayofweek"] = df["TransactionStartTime"].dt.dayofweek
        df["is_weekend"] = (df["txn_dayofweek"] >= 5).astype(int)

        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

        grouped = df.groupby("CustomerId", as_index=False)
        base = grouped.agg(
            total_txn_amount=("Amount", "sum"),
            avg_txn_amount=("Amount", "mean"),
            txn_count=("TransactionId", "count"),
            txn_amount_std=("Amount", "std"),
            total_value=("Value", "sum"),
            avg_value=("Value", "mean"),
            value_std=("Value", "std"),
            account_count=("AccountId", "nunique"),
            subscription_count=("SubscriptionId", "nunique"),
            provider_count=("ProviderId", "nunique"),
            product_count=("ProductId", "nunique"),
            product_category_count=("ProductCategory", "nunique"),
            channel_count=("ChannelId", "nunique"),
            country_code_count=("CountryCode", "nunique"),
            avg_txn_hour=("txn_hour", "mean"),
            avg_txn_day=("txn_day", "mean"),
            avg_txn_month=("txn_month", "mean"),
            avg_txn_year=("txn_year", "mean"),
            avg_txn_dayofweek=("txn_dayofweek", "mean"),
            weekend_txn_ratio=("is_weekend", "mean"),
            last_txn_time=("TransactionStartTime", "max"),
        )

        base["recency_days"] = (
            (snapshot_date - base["last_txn_time"]).dt.total_seconds() / 86400.0
        )
        base = base.drop(columns=["last_txn_time"])

        cat_part = grouped.agg(
            primary_currency=("CurrencyCode", _safe_mode),
            primary_channel=("ChannelId", _safe_mode),
            primary_product_category=("ProductCategory", _safe_mode),
            primary_pricing_strategy=("PricingStrategy", _safe_mode),
        )

        features = base.merge(cat_part, on="CustomerId", how="left")
        return features

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required raw input columns: {missing_cols}")


@dataclass
class ProcessedOutput:
    """Container for transformed features and fitted processing pipeline."""

    features: pd.DataFrame
    pipeline: Pipeline


def build_pipeline() -> Pipeline:
    """Create the single sklearn Pipeline object used for feature processing."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("customer_features", CustomerFeatureBuilder()),
            ("preprocessor", preprocessor),
        ]
    )


def fit_transform_to_dataframe(raw_df: pd.DataFrame) -> ProcessedOutput:
    """Fit the processing pipeline and return a model-ready DataFrame."""
    pipeline = build_pipeline()
    matrix = pipeline.fit_transform(raw_df)

    customer_features_df = pipeline.named_steps["customer_features"].transform(raw_df)
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_columns = preprocessor.get_feature_names_out().tolist()

    transformed_df = pd.DataFrame(matrix, columns=transformed_columns)
    transformed_df.insert(0, "CustomerId", customer_features_df["CustomerId"].values)
    return ProcessedOutput(features=transformed_df, pipeline=pipeline)


def process_raw_file(
    input_path: str,
    output_path: str,
    pipeline_path: str = "models/data_processing_pipeline.joblib",
) -> ProcessedOutput:
    """Load raw CSV, fit pipeline, and persist processed features + pipeline."""
    raw_df = pd.read_csv(input_path)
    result = fit_transform_to_dataframe(raw_df)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.features.to_csv(output_file, index=False)

    pipeline_file = Path(pipeline_path)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, pipeline_file)

    return result


def main() -> None:
    """Run processing with the default repository paths."""
    input_path = "data/raw/data.csv"
    output_path = "data/processed/model_features.csv"
    process_raw_file(input_path=input_path, output_path=output_path)
    print(f"Processed features written to {output_path}")


if __name__ == "__main__":
    main()
