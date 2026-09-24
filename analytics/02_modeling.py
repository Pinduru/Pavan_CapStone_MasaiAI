"""Train and evaluate classification and regression models on Titanic data."""

from __future__ import annotations

import json
from pathlib import Path

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
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
DATA_PATH = BASE_DIR / "titanic_clean.csv"
MODEL_PATH = BASE_DIR / "best_pipeline.joblib"
RANDOM_STATE = 42

CLASSIFICATION_FEATURES = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def classification_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_PATH)
    x = df[CLASSIFICATION_FEATURES]
    y = df["survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    fitted_models: dict[str, Pipeline] = {}
    metric_rows = []

    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(15, 4))
    plt.figure(figsize=(8, 6))
    for index, (name, estimator) in enumerate(estimators.items()):
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline
        metrics = classification_metrics(pipeline, x_test, y_test)
        metric_rows.append({"model": name, **metrics})

        predictions = pipeline.predict(x_test)
        matrix = confusion_matrix(y_test, predictions)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes_cm[index])
        axes_cm[index].set_title(name)
        axes_cm[index].set_xlabel("Predicted")
        axes_cm[index].set_ylabel("Actual")

        probabilities = pipeline.predict_proba(x_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={metrics['auc']:.3f})")

    fig_cm.tight_layout()
    fig_cm.savefig(CHART_DIR / "09_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Classifier ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "10_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    classification_table = pd.DataFrame(metric_rows).set_index("model")
    classification_table.to_csv(OUTPUT_DIR / "07_classification_metrics.csv")

    tree_pipeline = fitted_models["Decision Tree"]
    feature_names = tree_pipeline.named_steps["preprocessor"].get_feature_names_out()
    plt.figure(figsize=(24, 12))
    plot_tree(
        tree_pipeline.named_steps["model"],
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(CHART_DIR / "11_decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # The preprocessor is fitted only on x_train before the imbalance experiments.
    imbalance_preprocessor = make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    x_train_processed = imbalance_preprocessor.fit_transform(x_train)
    x_test_processed = imbalance_preprocessor.transform(x_test)
    imbalance_rows = []

    imbalance_models = {
        "Baseline": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Class weight balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    for name, model in imbalance_models.items():
        model.fit(x_train_processed, y_train)
        prediction = model.predict(x_test_processed)
        imbalance_rows.append(
            {
                "method": name,
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
            }
        )

    smote = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = smote.fit_resample(x_train_processed, y_train)
    smote_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    smote_model.fit(x_resampled, y_resampled)
    smote_prediction = smote_model.predict(x_test_processed)
    imbalance_rows.append(
        {
            "method": "SMOTE on training data",
            "precision": precision_score(y_test, smote_prediction, zero_division=0),
            "recall": recall_score(y_test, smote_prediction, zero_division=0),
            "f1": f1_score(y_test, smote_prediction, zero_division=0),
        }
    )
    imbalance_table = pd.DataFrame(imbalance_rows).set_index("method")
    imbalance_table.to_csv(OUTPUT_DIR / "08_imbalance_comparison.csv")

    rf_pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            (
                "model",
                RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE),
            ),
        ]
    )
    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [4, 6, None],
        "model__max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(rf_pipeline, parameter_grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(x_train, y_train)
    tuned_pipeline = search.best_estimator_
    tuned_metrics = classification_metrics(tuned_pipeline, x_test, y_test)
    oob_score = float(tuned_pipeline.named_steps["model"].oob_score_)

    regression_features = ["survived", "pclass", "age", "sibsp", "parch", "sex", "embarked"]
    regression_numeric = ["survived", "pclass", "age", "sibsp", "parch"]
    regression_categorical = ["sex", "embarked"]
    regression_x = df[regression_features]
    regression_y = df["fare"]
    reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(
        regression_x, regression_y, test_size=0.2, random_state=RANDOM_STATE
    )
    regression_pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(regression_numeric, regression_categorical)),
            ("model", LinearRegression()),
        ]
    )
    regression_pipeline.fit(reg_x_train, reg_y_train)
    fare_predictions = regression_pipeline.predict(reg_x_test)
    mae = mean_absolute_error(reg_y_test, fare_predictions)
    rmse = float(np.sqrt(mean_squared_error(reg_y_test, fare_predictions)))
    r_squared = r2_score(reg_y_test, fare_predictions)
    transformed_feature_count = len(
        regression_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    sample_count = len(reg_y_test)
    adjusted_r_squared = 1 - (1 - r_squared) * (sample_count - 1) / (
        sample_count - transformed_feature_count - 1
    )
    residuals = reg_y_test.to_numpy() - fare_predictions
    residual_spread_correlation = float(
        pd.Series(np.abs(residuals)).corr(pd.Series(fare_predictions), method="spearman")
    )
    heteroscedasticity = abs(residual_spread_correlation) >= 0.20

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fare_predictions, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Fare regression residual plot")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "12_regression_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    regression_metrics = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "mae": mae,
                "rmse": rmse,
                "r2": r_squared,
                "adjusted_r2": adjusted_r_squared,
            }
        ]
    ).set_index("model")
    regression_metrics.to_csv(OUTPUT_DIR / "09_regression_metrics.csv")

    best_name = classification_table["f1"].idxmax()
    best_pipeline = fitted_models[best_name]
    joblib.dump(best_pipeline, MODEL_PATH)
    reloaded_pipeline = joblib.load(MODEL_PATH)
    original_predictions = best_pipeline.predict(x_test.head(5))
    reloaded_predictions = reloaded_pipeline.predict(x_test.head(5))
    reload_match = bool(np.array_equal(original_predictions, reloaded_predictions))
    if not reload_match:
        raise RuntimeError("Reloaded pipeline predictions do not match.")

    comparison_table = pd.concat(
        {
            "Classification metrics": classification_table,
            "Regression metrics": regression_metrics,
        },
        axis=1,
    )
    comparison_table.to_csv(OUTPUT_DIR / "10_model_comparison.csv")

    results = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "class_percentages": {
            str(key): round(float(value * 100), 2)
            for key, value in y.value_counts(normalize=True).sort_index().items()
        },
        "classification_metrics": classification_table.round(4).to_dict(orient="index"),
        "imbalance_metrics": imbalance_table.round(4).to_dict(orient="index"),
        "best_parameters": search.best_params_,
        "grid_search_best_cv_f1": round(float(search.best_score_), 4),
        "tuned_random_forest_test_metrics": {key: round(value, 4) for key, value in tuned_metrics.items()},
        "oob_score": round(oob_score, 4),
        "regression_metrics": {
            "mae": round(float(mae), 4),
            "rmse": round(rmse, 4),
            "r2": round(float(r_squared), 4),
            "adjusted_r2": round(float(adjusted_r_squared), 4),
        },
        "residual_spread_correlation": round(residual_spread_correlation, 4),
        "heteroscedasticity_detected": heteroscedasticity,
        "saved_pipeline": best_name,
        "reload_predictions_match": reload_match,
    }
    (OUTPUT_DIR / "modeling_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best classifier by test F1: {best_name}")
    print(f"Saved and reloaded pipeline successfully: {reload_match}")
    print("Modeling completed successfully.")


if __name__ == "__main__":
    main()
