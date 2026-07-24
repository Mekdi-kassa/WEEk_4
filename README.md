# Credit Risk Probability Model for Alternative Data

This repository contains an end-to-end implementation for building, deploying, and automating a credit risk model for buy-now-pay-later decisions using alternative transaction data.

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Credit Scoring Business Understanding

### 1) Basel II and Why Interpretability Matters
Basel II emphasizes disciplined risk measurement, transparent governance, and auditable model use in credit decisions. In practice, this means model developers must be able to explain what signals drive risk predictions, prove that development and validation procedures are repeatable, and document assumptions and limitations. An interpretable and well-documented model supports:

- Regulatory review and internal model risk management.
- Traceability from input features to prediction outcomes.
- Ongoing monitoring for drift, performance degradation, and bias.
- Defensible adverse-action style explanations for business stakeholders.

For this project, Basel II expectations influence the workflow by requiring reproducible pipelines, fixed random seeds, clear feature lineage, experiment tracking, and model comparison logs.

### 2) Why a Proxy Default Variable Is Necessary
The dataset does not contain an explicit loan-default label, so supervised credit risk modeling cannot proceed directly. A proxy target is required to approximate default risk using observed behavior. Here, customer engagement and payment behavior signals (for example RFM patterns) are used to define likely high-risk versus low-risk groups.

Key business risks introduced by proxy-based prediction include:

- Label noise: proxy labels are imperfect and can misclassify true risk.
- Concept mismatch: disengagement or low activity may not always imply inability or unwillingness to repay.
- Fairness and segment bias risks if proxy behavior is uneven across customer groups.
- Operational risk if credit policy over-relies on proxy outputs without conservative controls.

Mitigations include explicit documentation of proxy assumptions, periodic back-testing once true repayment outcomes become available, policy guardrails, and threshold tuning with risk-team oversight.

### 3) Interpretable vs High-Performance Models in Regulated Finance
A simple model (for example Logistic Regression with WoE features) typically provides strong transparency: coefficients are explainable, monotonic relationships are easier to enforce, and governance is simpler. However, predictive power may be lower when relationships are nonlinear or interaction-heavy.

A high-performance model (for example Gradient Boosting) often improves discrimination and ranking power but can be less transparent and harder to govern. In regulated contexts, this creates trade-offs in explainability, validation effort, and model risk controls.

A practical strategy is champion-challenger modeling:

- Use an interpretable baseline as the governance anchor.
- Evaluate stronger nonlinear models as challengers.
- Select based on both predictive performance and governance readiness.
- Add explainability tooling and robust monitoring if adopting a complex model.

## Task Workflow

- Main branch: repository structure and shared baseline.
- task-1: business understanding and regulatory framing.
- task-2: exploratory data analysis in notebook.
- task-3: feature engineering pipeline.
- task-4: proxy target engineering with RFM clustering.
- task-5: model training, tuning, MLflow tracking, and tests.
- task-6: API deployment, Docker, and CI/CD hardening.
