"""Tests for data processing helpers."""

from __future__ import annotations

import pandas as pd

from src.data_processing import build_rfm_high_risk_labels
from src.data_processing import fit_transform_to_dataframe


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "BatchId": ["b1", "b1", "b2", "b3", "b4", "b4"],
            "AccountId": ["a1", "a1", "a2", "a3", "a2", "a3"],
            "SubscriptionId": ["s1", "s1", "s2", "s3", "s2", "s3"],
            "CustomerId": ["c1", "c1", "c2", "c3", "c2", "c3"],
            "CurrencyCode": ["UGX", "UGX", "UGX", "UGX", "UGX", "UGX"],
            "CountryCode": [256, 256, 256, 256, 256, 256],
            "ProviderId": ["p1", "p1", "p2", "p2", "p1", "p3"],
            "ProductId": ["pr1", "pr2", "pr2", "pr3", "pr4", "pr3"],
            "ProductCategory": [
                "electronics",
                "electronics",
                "fashion",
                "fashion",
                "grocery",
                "fashion",
            ],
            "ChannelId": ["web", "web", "android", "ios", "android", "ios"],
            "Amount": [100.0, 150.0, 50.0, 30.0, 70.0, 20.0],
            "Value": [100.0, 150.0, 50.0, 30.0, 70.0, 20.0],
            "TransactionStartTime": [
                "2024-01-01 10:00:00",
                "2024-01-15 12:00:00",
                "2024-01-03 08:30:00",
                "2024-01-10 14:00:00",
                "2024-02-01 09:30:00",
                "2024-02-10 18:45:00",
            ],
            "PricingStrategy": [1, 1, 2, 2, 1, 3],
            "FraudResult": [0, 0, 0, 0, 0, 0],
        }
    )


def test_fit_transform_includes_customer_id_and_numeric_matrix() -> None:
    raw = _sample_transactions()
    result = fit_transform_to_dataframe(raw_df=raw, include_target=False)

    assert "CustomerId" in result.features.columns
    assert result.features.shape[0] == raw["CustomerId"].nunique()
    assert result.features.drop(columns=["CustomerId"]).shape[1] > 0


def test_rfm_high_risk_labels_are_binary_and_complete() -> None:
    raw = _sample_transactions()
    labels = build_rfm_high_risk_labels(raw, n_clusters=3, random_state=42)

    assert set(labels.columns) == {"CustomerId", "is_high_risk"}
    assert labels.shape[0] == raw["CustomerId"].nunique()
    assert set(labels["is_high_risk"].unique()).issubset({0, 1})
