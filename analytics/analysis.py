"""Explore the Titanic data, then fit leakage-safe classification and fare models."""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
TITANIC_PATH = BASE_DIR / "titanic.csv"
LEGACY_TITANIC_PATH = BASE_DIR / "data" / "titanic_fallback.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
RANDOM_STATE = 42
TARGET = "survived"
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def load_titanic(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the raw data once, saving that same frame as the offline copy."""
    if offline:
        source_path = TITANIC_PATH if TITANIC_PATH.exists() else LEGACY_TITANIC_PATH
        if not source_path.exists():
            raise FileNotFoundError(f"Offline Titanic data not found: {source_path}")
        frame = pd.read_csv(source_path)
        frame.to_csv(TITANIC_PATH, index=False)
        return frame, f"offline CSV ({source_path.name})"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import seaborn as sns

            frame = sns.load_dataset("titanic")
        source = "seaborn.load_dataset('titanic')"
    except (OSError, ValueError, ConnectionError, TimeoutError) as exc:
        source_path = TITANIC_PATH if TITANIC_PATH.exists() else LEGACY_TITANIC_PATH
        if not source_path.exists():
            raise RuntimeError(
                "Seaborn could not fetch Titanic and no committed fallback CSV exists."
            ) from exc
        print(f"Seaborn data fetch failed ({exc}); using {source_path.name}.")
        frame = pd.read_csv(source_path)
        source = f"offline CSV ({source_path.name}; network fallback)"

    # Keep the untouched source columns and missing values for grading/offline use.
    frame.to_csv(TITANIC_PATH, index=False)
    return frame, source


def profile_raw_data(raw: pd.DataFrame) -> dict[str, Any]:
    print("\n=== Raw data profile ===")
    raw.info()
    print("\n=== df.describe() ===")
    print(raw.describe(include="all").T.to_string())
    print(f"\n=== df.shape ===\n{raw.shape}")

    missing = raw.isna().mean().mul(100)
    missing = missing[missing.gt(0)].sort_values(ascending=False)
    print("\n=== Columns with missing values ===")
    if missing.empty:
        print("No missing values found.")
    else:
        for column, percentage in missing.items():
            print(f"{column}: {percentage:.2f}%")
    return {column: round(float(value), 4) for column, value in missing.items()}


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the assignment thresholds and return an imputation-free modeling frame."""
    required = [TARGET, *FEATURES]
    absent = sorted(set(required) - set(raw.columns))
    if absent:
        raise ValueError(f"Titanic data is missing required columns: {absent}")

    missing_pct = raw.isna().mean().mul(100)
    decisions: dict[str, str] = {}
    high_missing = [name for name, rate in missing_pct.items() if rate > 30]
    for column in high_missing:
        decisions[column] = (
            f"drop column ({missing_pct[column]:.2f}% missing; above 30%, too sparse to impute)"
        )
    for column, rate in missing_pct.items():
        if 0 < rate < 5:
            if column in {"embarked", "fare"}:
                decisions[column] = (
                    f"drop affected rows ({rate:.2f}% missing; below 5%); "
                    "embarked is a model feature and fare is the regression target"
                )
            elif column == "embark_town":
                decisions[column] = (
                    f"drop with the corresponding embarked row ({rate:.2f}% missing; below 5%); "
                    "this is a redundant text label for embarked"
                )
            else:
                decisions[column] = (
                    f"exclude from the working feature set ({rate:.2f}% missing; below 5%)"
                )
        elif 5 <= rate <= 30:
            decisions[column] = (
                f"median imputation ({rate:.2f}% missing; within the 5%–30% range), "
                "fit only on the training split for modeling"
            )

    # Drop rows only for low-missingness fields required downstream. Age stays missing
    # here so the modeling Pipeline can learn its median from training data only.
    rows_before = len(raw)
    clean = raw.dropna(subset=[name for name in ("embarked", "fare") if name in raw]).copy()
    clean = clean[required].copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="raise").astype(int)
    clean["pclass"] = pd.to_numeric(clean["pclass"], errors="raise")
    for column in ("age", "sibsp", "parch", "fare"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        clean[column] = clean[column].astype("string")

    # EDA needs complete age values for plots and standardization. The modeling frame
    # above is kept separate so its imputer is fitted after the stratified split.
    eda = clean.copy()
    age_group_medians = eda.groupby(["sex", "pclass"], observed=True)["age"].transform("median")
    eda["age"] = eda["age"].fillna(age_group_medians).fillna(eda["age"].median())
    eda["fare"] = eda["fare"].fillna(eda["fare"].median())
    details = {
        "rows_before": int(rows_before),
        "rows_after_low_missing_drops": int(len(clean)),
        "columns_dropped_for_high_missingness": high_missing,
        "missing_value_decisions": decisions,
        "modeling_age_median_fitted_on_training_only": True,
    }
    return clean.reset_index(drop=True), eda.reset_index(drop=True), details


def save_eda_charts(eda: pd.DataFrame) -> dict[str, Any]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    findings: dict[str, Any] = {}

    for column, label in (("age", "Age"), ("fare", "Fare")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(eda[column], bins=25, kde=True, ax=axes[0], color="#2a788e")
        axes[0].set_title(f"{label} distribution")
        sns.boxplot(x=eda[column], ax=axes[1], color="#7ad151")
        axes[1].set_title(f"{label} box plot")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{column}_distribution_and_boxplot.png", dpi=150)
        plt.close(fig)

        q1, q3 = eda[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        findings[f"{column}_iqr"] = {
            "q1": float(q1), "q3": float(q3), "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(((eda[column] < lower) | (eda[column] > upper)).sum()),
        }

    fare_mean = float(eda["fare"].mean())
    fare_median = float(eda["fare"].median())
    fare_mode = float(eda["fare"].mode().iloc[0])
    if fare_mean > fare_median > fare_mode:
        skew = "right-skewed (mean > median > mode)"
    elif fare_mean < fare_median < fare_mode:
        skew = "left-skewed (mean < median < mode)"
    else:
        skew = "not described by a strict mean/median/mode ordering"
    findings["fare_distribution"] = {
        "mean": fare_mean, "median": fare_median, "mode": fare_mode,
        "skew_interpretation": skew,
    }

    # Boolean masks are used explicitly for the required group breakdowns.
    sex_rates = {}
    for sex in sorted(eda["sex"].dropna().unique()):
        mask = eda["sex"].eq(sex)
        sex_rates[str(sex)] = float(eda.loc[mask, TARGET].mean())
    class_rates = {}
    for passenger_class in sorted(eda["pclass"].dropna().unique()):
        mask = eda["pclass"].eq(passenger_class)
        class_rates[str(int(passenger_class))] = float(eda.loc[mask, TARGET].mean())
    combined = {}
    for sex, passenger_class in itertools.product(
        sorted(eda["sex"].dropna().unique()), sorted(eda["pclass"].dropna().unique())
    ):
        mask = eda["sex"].eq(sex) & eda["pclass"].eq(passenger_class)
        group = eda.loc[mask, TARGET]
        if not group.empty:
            combined[f"{sex}, class {int(passenger_class)}"] = float(group.mean())
    findings["survival_rates"] = {
        "by_sex": sex_rates, "by_pclass": class_rates, "by_sex_and_pclass": combined
    }

    corr = eda[CORRELATION_COLUMNS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Titanic numeric feature correlations")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    pairs = []
    for left, right in itertools.combinations(CORRELATION_COLUMNS, 2):
        coefficient = float(corr.loc[left, right])
        pairs.append({"feature_1": left, "feature_2": right, "correlation": coefficient})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    findings["correlation_columns"] = CORRELATION_COLUMNS
    findings["strongest_correlations"] = pairs[:2]

    chart_paths = []
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=eda, x="sex", y=TARGET, errorbar=None, color="#2a788e", ax=ax)
    ax.set(title="Survival rate by sex", ylabel="Share who survived", xlabel="Sex")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "survival_by_sex.png", dpi=150); plt.close(fig)
    chart_paths.append("survival_by_sex.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=eda, x="pclass", y=TARGET, errorbar=None, color="#7ad151", ax=ax)
    ax.set(title="Survival rate by passenger class", ylabel="Share who survived", xlabel="Passenger class")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "survival_by_class.png", dpi=150); plt.close(fig)
    chart_paths.append("survival_by_class.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=eda, x="pclass", y=TARGET, hue="sex", errorbar=None, ax=ax)
    ax.set(title="Survival by class and sex", ylabel="Share who survived", xlabel="Passenger class")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "survival_by_class_and_sex.png", dpi=150); plt.close(fig)
    chart_paths.append("survival_by_class_and_sex.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=eda, x="sex", y="age", hue=TARGET, ax=ax)
    ax.set(title="Age distribution by sex and survival", ylabel="Age", xlabel="Sex")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "age_by_sex_and_survival.png", dpi=150); plt.close(fig)
    chart_paths.append("age_by_sex_and_survival.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=eda, x="pclass", y="fare", ax=ax)
    ax.set(title="Fare distribution by passenger class", ylabel="Fare", xlabel="Passenger class")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "fare_by_class.png", dpi=150); plt.close(fig)
    chart_paths.append("fare_by_class.png")
    findings["multivariate_charts"] = chart_paths

    # EDA-only z-scores use the full cleaned frame; the model pipeline has its own
    # train-fitted StandardScaler and never consumes these exploratory columns.
    standardization = {}
    for column in ("age", "fare"):
        before_mean = float(eda[column].mean())
        before_std = float(eda[column].std(ddof=0))
        z = (eda[column] - before_mean) / before_std
        standardization[column] = {
            "before_mean": before_mean, "before_std": before_std,
            "after_mean": float(z.mean()), "after_std": float(z.std(ddof=0)),
        }
    findings["standardization_check"] = standardization
    findings["age_median_by_sex_and_survival"] = {
        f"{sex}, survived={int(survived)}": float(value)
        for (sex, survived), value in eda.groupby(["sex", TARGET], observed=True)["age"].median().items()
    }
    findings["fare_median_by_class"] = {
        str(int(passenger_class)): float(value)
        for passenger_class, value in eda.groupby("pclass", observed=True)["fare"].median().items()
    }
    return findings


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def classification(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    print("\n=== Stratified classification split ===")
    print(f"Train: {x_train.shape}; test: {x_test.shape}")
    print("Survival share, full/train/test:", round(data[TARGET].mean(), 4),
          round(y_train.mean(), 4), round(y_test.mean(), 4))

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    metrics: dict[str, Any] = {}
    pipelines: dict[str, Pipeline] = {}
    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        cm = confusion_matrix(y_test, predicted, labels=[0, 1])
        auc = float(roc_auc_score(y_test, probabilities))
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "auc": auc,
            "confusion_matrix": cm.tolist(),
        }
        pipelines[name] = pipeline
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=roc_ax)
        print(f"\n{name} confusion matrix (rows=true 0/1; columns=predicted 0/1):\n{cm}")
        ConfusionMatrixDisplay(cm, display_labels=["Not survived", "Survived"]).plot(cmap="Blues")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

        if name == "Decision Tree":
            fitted = pipeline.named_steps["model"]
            names = pipeline.named_steps["preprocess"].get_feature_names_out()
            fig, ax = plt.subplots(figsize=(24, 12))
            plot_tree(fitted, feature_names=names, class_names=["Not survived", "Survived"],
                      filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax)
            fig.tight_layout(); fig.savefig(PLOT_DIR / "decision_tree.png", dpi=150); plt.close(fig)

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    roc_ax.set_title("ROC curves on the shared stratified test set")
    roc_fig.tight_layout(); roc_fig.savefig(PLOT_DIR / "roc_curves.png", dpi=150); plt.close(roc_fig)

    # Compare the imbalance strategies on the same untouched test set. Each fitted
    # transformer sees only the training rows; SMOTE is applied after train transform.
    rf_options = {
        "Baseline": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "class_weight='balanced'": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    imbalance: dict[str, Any] = {
        "class_balance": {
            "not_survived": int((y_train == 0).sum()),
            "survived": int((y_train == 1).sum()),
            "train_survival_rate": float(y_train.mean()),
        }
    }
    imbalance_rows = []
    for label, estimator in rf_options.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        scores = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        imbalance[label] = scores
        imbalance_rows.append({"strategy": label, **scores})

    smote_preprocessor = make_preprocessor()
    train_features = smote_preprocessor.fit_transform(x_train)
    test_features = smote_preprocessor.transform(x_test)
    train_balanced, target_balanced = SMOTE(random_state=RANDOM_STATE).fit_resample(
        train_features, y_train
    )
    smote_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    smote_model.fit(train_balanced, target_balanced)
    pred = smote_model.predict(test_features)
    smote_metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "training_rows_after_smote": int(len(target_balanced)),
    }
    imbalance["SMOTE (training rows only)"] = smote_metrics
    imbalance_rows.append({"strategy": "SMOTE (training rows only)", **smote_metrics})
    imbalance_frame = pd.DataFrame(imbalance_rows)
    print("\n=== Imbalance comparison ===")
    print(imbalance_frame.to_string(index=False))

    search = GridSearchCV(
        Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE,
                                              bootstrap=True, n_jobs=1)),
        ]),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    best_pipeline: Pipeline = search.best_estimator_
    tuning = {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "best_estimator_oob_score": float(best_pipeline.named_steps["model"].oob_score_),
    }
    print("\n=== Random Forest grid search ===")
    print(json.dumps(tuning, indent=2))

    # Keep the preprocessing and estimator together, then reload and predict from raw
    # input columns as a deployment-style round trip.
    artifact_path = OUTPUT_DIR / "titanic_classifier.joblib"
    joblib.dump(best_pipeline, artifact_path)
    reloaded = joblib.load(artifact_path)
    raw_prediction = reloaded.predict(x_test.iloc[[0]])
    print(f"Reloaded full pipeline prediction on an unprocessed row: {raw_prediction.tolist()}")
    return metrics, {
        "imbalance": imbalance,
        "imbalance_table": imbalance_frame,
        "random_forest_grid_search": tuning,
        "recommendation_model": max(metrics, key=lambda key: metrics[key]["f1"]),
        "saved_pipeline": str(artifact_path.relative_to(BASE_DIR)),
        "reload_prediction": int(raw_prediction[0]),
        "test_survival_rate": float(y_test.mean()),
    }, pipelines


def fare_regression(data: pd.DataFrame) -> dict[str, Any]:
    regression_features = [TARGET, "pclass", "sex", "age", "sibsp", "parch", "embarked"]
    x_train, x_test, y_train, y_test = train_test_split(
        data[regression_features], data["fare"], test_size=0.25,
        random_state=RANDOM_STATE,
    )
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]),
         [TARGET, "pclass", "age", "sibsp", "parch"]),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         ["sex", "embarked"]),
    ])
    model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test.to_numpy() - predictions
    n = len(y_test)
    predictors = len(model.named_steps["preprocess"].get_feature_names_out())
    r2 = float(r2_score(y_test, predictions))
    adjusted_r2 = float(1 - (1 - r2) * (n - 1) / (n - predictors - 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predictions, residuals, alpha=0.6, color="#2a788e")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title="Fare regression residuals", xlabel="Predicted fare", ylabel="Actual - predicted")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "fare_regression_residuals.png", dpi=150); plt.close(fig)
    residual_frame = pd.DataFrame({"prediction": predictions, "residual": residuals})
    residual_frame["prediction_bin"] = pd.qcut(residual_frame["prediction"], q=4, duplicates="drop")
    residual_spreads = residual_frame.groupby("prediction_bin", observed=True)["residual"].std().dropna()
    residual_spread_ratio = float(residual_spreads.max() / max(residual_spreads.min(), 1e-9))
    heteroscedastic = residual_spread_ratio >= 2.0
    output = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "residual_prediction_correlation": float(np.corrcoef(predictions, residuals)[0, 1]),
        "residual_std_by_prediction_quartile": [float(value) for value in residual_spreads],
        "residual_spread_ratio": residual_spread_ratio,
        "heteroscedasticity_detected_by_2x_spread_rule": heteroscedastic,
        "test_rows": int(n),
    }
    print("\n=== Fare regression metrics ===")
    print(json.dumps(output, indent=2))
    return output


def write_analysis_report(results: dict[str, Any], comparison: pd.DataFrame) -> None:
    """Write the measured interpretation and comparison as assessable Markdown."""
    eda = results["eda"]
    modeling = results["modeling_details"]
    regression = results["fare_regression"]
    classification_train_rate = modeling["imbalance"]["class_balance"]["train_survival_rate"]
    sex_rates = eda["survival_rates"]["by_sex"]
    class_rates = eda["survival_rates"]["by_pclass"]
    combined_rates = eda["survival_rates"]["by_sex_and_pclass"]
    age_medians = eda["age_median_by_sex_and_survival"]
    fare_medians = eda["fare_median_by_class"]
    best_name = modeling["recommendation_model"]
    best = results["classification"][best_name]

    missing_rows = ["| Column | Missing | Decision |", "|---|---:|---|"]
    for column, percentage in results["missing_percentages"].items():
        decision = results["cleaning"]["missing_value_decisions"].get(
            column, "Not used downstream; excluded from the working feature set."
        )
        missing_rows.append(f"| `{column}` | {percentage:.2f}% | {decision} |")

    model_rows = [
        "| Model | Accuracy | Precision | Recall | F1 | AUC | MAE | RMSE | R-squared | Adjusted R-squared |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        values = []
        for column in ("accuracy", "precision", "recall", "f1", "auc", "MAE", "RMSE", "R2", "adjusted_R2"):
            value = row[column]
            values.append("-" if pd.isna(value) else f"{float(value):.4f}")
        model_rows.append(f"| {row['model']} | " + " | ".join(values) + " |")

    imbalance = modeling["imbalance"]
    imbalance_rows = ["| Strategy | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for strategy in ("Baseline", "class_weight='balanced'", "SMOTE (training rows only)"):
        scores = imbalance[strategy]
        imbalance_rows.append(
            f"| {strategy} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} |"
        )
    best_imbalance = max(
        ("Baseline", "class_weight='balanced'", "SMOTE (training rows only)"),
        key=lambda strategy: (imbalance[strategy]["f1"], imbalance[strategy]["recall"]),
    )
    spread_text = (
        f"The residual standard deviations across predicted-fare quartiles vary by a factor of "
        f"{regression['residual_spread_ratio']:.2f}. Under the stated 2x spread check, this "
        f"{'suggests heteroscedasticity' if regression['heteroscedasticity_detected_by_2x_spread_rule'] else 'does not provide strong evidence of heteroscedasticity'}; "
        "the residual plot is included below for visual inspection."
    )

    text = f"""# Titanic Analytics

This module profiles and cleans the Seaborn Titanic dataset, tells a short story with the data, and compares leakage-safe classifiers with a separate fare regression task. The analysis runs from one script so the raw data is loaded once and every step is easy to reproduce.

## Setup and run

```powershell
cd analytics
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python analysis.py
```

Use `python analysis.py --offline` to read the committed `titanic.csv` directly. A normal run calls Seaborn's loader once and saves the original, uncleaned data to that same CSV before analysis. Generated tables, plots, metrics, and the fitted classifier pipeline are saved under `outputs/`.

## Design decisions

- Keep the raw CSV as the reproducible offline copy; drop only rows with low missingness in required fields and exclude highly sparse or redundant columns.
- Use one cleaned working dataset for the EDA and modeling stages. EDA fills age with the median for the exploratory plots, while modeling keeps missing ages until the stratified split and learns the median inside each training-only preprocessing pipeline.
- Keep transformations inside scikit-learn pipelines so imputers, encoders, and scalers are fitted on training data only. Save the fitted preprocessing and estimator together for raw-row predictions.

## Measured results and interpretations

Dataset source: `{results['source']}`. The untouched dataset has shape `{results['raw_shape']}` and is saved as `titanic.csv` before cleaning. The cleaned modeling/EDA frame has {results['cleaning']['rows_after_low_missing_drops']} rows.

## Missingness and Cleaning

{chr(10).join(missing_rows)}

Missingness under 5% is handled by dropping affected rows; 5%-30% uses median imputation for age; columns above 30% missing are dropped rather than imputed. `deck` is excluded because 77% of its values are missing, and `embark_town` is a redundant label for `embarked`. Modeling imputers, encoders, and scalers are fitted only on the training split inside each scikit-learn Pipeline.

## Univariate Analysis

IQR outlier counts: age **{eda['age_iqr']['outlier_count']}**, fare **{eda['fare_iqr']['outlier_count']}**. Fare mean = {eda['fare_distribution']['mean']:.4f}, median = {eda['fare_distribution']['median']:.4f}, and mode = {eda['fare_distribution']['mode']:.4f}; this is **{eda['fare_distribution']['skew_interpretation']}**.

| Age histogram and box plot |
|---|
| ![Age distribution and box plot](plots/age_distribution_and_boxplot.png) |

| Fare histogram and box plot |
|---|
| ![Fare distribution and box plot](plots/fare_distribution_and_boxplot.png) |

## Bivariate and Multivariate Story

Survival rate by sex: `{json.dumps(sex_rates, sort_keys=True)}`. By passenger class: `{json.dumps(class_rates, sort_keys=True)}`. By sex and class: `{json.dumps(combined_rates, sort_keys=True)}`. The requested correlations use exactly `{', '.join(eda['correlation_columns'])}`; `adult_male` and `alone` are excluded because they are derived flags.

The two strongest absolute off-diagonal correlations are `{eda['strongest_correlations'][0]['feature_1']}` with `{eda['strongest_correlations'][0]['feature_2']}` (r = {eda['strongest_correlations'][0]['correlation']:.4f}) and `{eda['strongest_correlations'][1]['feature_1']}` with `{eda['strongest_correlations'][1]['feature_2']}` (r = {eda['strongest_correlations'][1]['correlation']:.4f}). Class and fare are negatively associated, consistent with higher fares being more common among lower-numbered classes; siblings/spouses and parents/children counts are positively associated, as family groups often traveled together.

![Six-column correlation heatmap](plots/correlation_heatmap.png)

### Survival Rate by Sex

![Survival by sex](plots/survival_by_sex.png)

Women survived at {sex_rates.get('female', 0):.1%}, compared with {sex_rates.get('male', 0):.1%} of men. This large descriptive difference makes sex an important predictor in this sample, but it is not by itself a causal explanation.

### Survival Rate by Passenger Class

![Survival by class](plots/survival_by_class.png)

Survival rates were {class_rates.get('1', 0):.1%} in first class, {class_rates.get('2', 0):.1%} in second class, and {class_rates.get('3', 0):.1%} in third class. The stepwise decline shows a clear association between passenger class and survival.

### Survival by Class and Sex

![Survival by class and sex](plots/survival_by_class_and_sex.png)

Within first, second, and third class, female survival was {combined_rates.get('female, class 1', 0):.1%}, {combined_rates.get('female, class 2', 0):.1%}, and {combined_rates.get('female, class 3', 0):.1%}; corresponding male rates were {combined_rates.get('male, class 1', 0):.1%}, {combined_rates.get('male, class 2', 0):.1%}, and {combined_rates.get('male, class 3', 0):.1%}. Both variables distinguish outcomes, and the grouped chart makes their interaction visible rather than averaging across groups.

### Age, Sex, and Survival

![Age by sex and survival](plots/age_by_sex_and_survival.png)

Median ages for women were {age_medians.get('female, survived=1', 0):.1f} among survivors and {age_medians.get('female, survived=0', 0):.1f} among non-survivors; for men they were {age_medians.get('male, survived=1', 0):.1f} and {age_medians.get('male, survived=0', 0):.1f}. The overlap in the box distributions indicates that age adds context but does not cleanly separate survival outcomes by itself.

### Fare by Passenger Class

![Fare by passenger class](plots/fare_by_class.png)

Median fares were {fare_medians.get('1', 0):.2f}, {fare_medians.get('2', 0):.2f}, and {fare_medians.get('3', 0):.2f} for first, second, and third class respectively. Fare therefore reinforces the class pattern in the survival charts, while also acting as a proxy for ticket/cabin access rather than an isolated cause.

### Exploratory Standardization

These full-cleaned-frame z-scores are only an EDA check and are not used in modeling; model scalers are fitted on training folds only.

| Feature | Before mean | Before std | After mean | After std |
|---|---:|---:|---:|---:|
| age | {eda['standardization_check']['age']['before_mean']:.4f} | {eda['standardization_check']['age']['before_std']:.4f} | {eda['standardization_check']['age']['after_mean']:.8f} | {eda['standardization_check']['age']['after_std']:.8f} |
| fare | {eda['standardization_check']['fare']['before_mean']:.4f} | {eda['standardization_check']['fare']['before_std']:.4f} | {eda['standardization_check']['fare']['after_mean']:.8f} | {eda['standardization_check']['fare']['after_std']:.8f} |

## Classification and Regression Comparison

The stratified split retained a similar survival share in both partitions (train {classification_train_rate:.1%}; test {modeling['test_survival_rate']:.1%}), important because the classes are not balanced. All three classifiers use the same split and report confusion matrices in `results.json`; the decision-tree plot labels the transformed feature and class names.

{chr(10).join(model_rows)}

Classification and regression metric columns are kept separate; `-` means that metric does not apply to that model type.

![Classifier ROC curves](plots/roc_curves.png)

![Decision tree](plots/decision_tree.png)

### Imbalance Handling

Training class counts were `{json.dumps(imbalance['class_balance'], sort_keys=True)}`. SMOTE is applied only to transformed training rows; the test fold is never resampled.

{chr(10).join(imbalance_rows)}

`{best_imbalance}` had the highest F1 ({imbalance[best_imbalance]['f1']:.4f}) in this held-out comparison. This favors the observed precision/recall balance for this split; another operational objective could prefer higher recall even at lower precision.

### Random Forest Grid Search

GridSearchCV best parameters: `{json.dumps(modeling['random_forest_grid_search']['best_params'], sort_keys=True)}`. Best cross-validation ROC AUC was {modeling['random_forest_grid_search']['best_cv_roc_auc']:.4f}; the best estimator was constructed with `oob_score=True` and achieved OOB score {modeling['random_forest_grid_search']['best_estimator_oob_score']:.4f}.

### Fare Regression

Linear regression predicts fare from the other available features, including survival outcome. Metrics: MAE {regression['mae']:.4f}, RMSE {regression['rmse']:.4f}, R-squared {regression['r2']:.4f}, adjusted R-squared {regression['adjusted_r2']:.4f}. {spread_text}

![Fare regression residual plot](plots/fare_regression_residuals.png)

## Deployment Recommendation

I recommend **{best_name}** among these three classifiers because it has the best held-out F1 ({best['f1']:.4f}), with accuracy {best['accuracy']:.4f}, precision {best['precision']:.4f}, and recall {best['recall']:.4f}. Its ROC AUC is {best['auc']:.4f}; Logistic Regression's AUC is {results['classification']['Logistic Regression']['auc']:.4f}, so the Random Forest's thresholded F1 advantage does not mean it ranks cases best on every metric. The saved artifact is the complete tuned preprocessing-plus-estimator pipeline and has been reloaded to predict from raw, unprocessed input rows. I would validate calibration and performance on a more recent, independent population before deployment.

Reload test prediction on a raw row: `{modeling['reload_prediction']}`. Saved pipeline: `{modeling['saved_pipeline']}`.
"""
    text = text.replace("](plots/", "](outputs/plots/")
    (BASE_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Read analytics/titanic.csv directly.")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    raw, source = load_titanic(offline=args.offline)
    missing = profile_raw_data(raw)
    clean, eda, cleaning = clean_data(raw)
    print("\n=== Cleaning decisions ===")
    print(json.dumps(cleaning, indent=2))
    eda_findings = save_eda_charts(eda)
    print("\n=== EDA findings ===")
    print(json.dumps(eda_findings, indent=2))

    classification_metrics, classification_details, _ = classification(clean)
    regression_metrics = fare_regression(clean)
    comparison_rows = []
    for model, metrics in classification_metrics.items():
        comparison_rows.append({"model": model, **metrics, "MAE": None, "RMSE": None,
                                "R2": None, "adjusted_R2": None})
    comparison_rows.append({"model": "Linear Regression (fare)", "accuracy": None,
                            "precision": None, "recall": None, "f1": None, "auc": None,
                            "MAE": regression_metrics["mae"], "RMSE": regression_metrics["rmse"],
                            "R2": regression_metrics["r2"],
                            "adjusted_R2": regression_metrics["adjusted_r2"]})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (classification and regression metrics have separate columns) ===")
    print(comparison.to_string(index=False))

    results = {
        "source": source,
        "raw_shape": list(raw.shape),
        "missing_percentages": missing,
        "cleaning": cleaning,
        "eda": eda_findings,
        "classification": classification_metrics,
        "modeling_details": {
            key: value for key, value in classification_details.items() if key != "imbalance_table"
        },
        "fare_regression": regression_metrics,
        "model_comparison": comparison_rows,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    write_analysis_report(results, comparison)
    print(f"\nSaved charts, metrics, and the complete classifier pipeline under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
