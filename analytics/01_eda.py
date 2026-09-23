"""Load the Titanic data once, clean it and create the EDA outputs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
RAW_DATA_PATH = BASE_DIR / "titanic.csv"
CLEAN_DATA_PATH = BASE_DIR / "titanic_clean.csv"


def save_chart(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def iqr_outlier_count(series: pd.Series) -> tuple[int, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return count, float(lower), float(upper)


def strongest_correlations(correlation: pd.DataFrame) -> list[dict[str, float | str]]:
    pairs = []
    columns = correlation.columns.tolist()
    for first_index, first in enumerate(columns):
        for second in columns[first_index + 1 :]:
            value = float(correlation.loc[first, second])
            pairs.append({"first": first, "second": second, "correlation": value})
    return sorted(pairs, key=lambda item: abs(item["correlation"]), reverse=True)[:2]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # This is the only network/cache load in the whole analytics module.
    df = sns.load_dataset("titanic")
    df.to_csv(RAW_DATA_PATH, index=False)

    info_buffer = io.StringIO()
    df.info(buf=info_buffer)
    missing_percentages = (df.isna().mean() * 100).loc[lambda values: values > 0]

    profile_text = [
        "DATASET SHAPE",
        str(df.shape),
        "",
        "DATAFRAME INFO",
        info_buffer.getvalue(),
        "DATAFRAME DESCRIPTION",
        df.describe(include="all").to_string(),
        "",
        "MISSING VALUES (%)",
        missing_percentages.round(2).to_string(),
    ]
    (OUTPUT_DIR / "01_profile.txt").write_text("\n".join(profile_text), encoding="utf-8")

    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=["embarked", "embark_town"]).copy()
    clean_df["age"] = clean_df["age"].fillna(clean_df["age"].median())
    clean_df = clean_df.drop(columns=["deck"])
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    age_outliers, age_lower, age_upper = iqr_outlier_count(clean_df["age"])
    fare_outliers, fare_lower, fare_upper = iqr_outlier_count(clean_df["fare"])
    fare_mean = float(clean_df["fare"].mean())
    fare_median = float(clean_df["fare"].median())
    fare_mode = float(clean_df["fare"].mode().iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(clean_df["age"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Age distribution")
    sns.boxplot(x=clean_df["age"], ax=axes[0, 1])
    axes[0, 1].set_title("Age box plot")
    sns.histplot(clean_df["fare"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Fare distribution")
    sns.boxplot(x=clean_df["fare"], ax=axes[1, 1])
    axes[1, 1].set_title("Fare box plot")
    save_chart("01_age_fare_univariate.png")

    female_mask = clean_df["sex"] == "female"
    male_mask = clean_df["sex"] == "male"
    survival_by_sex = pd.DataFrame(
        {
            "sex": ["female", "male"],
            "survival_rate": [
                clean_df.loc[female_mask, "survived"].mean(),
                clean_df.loc[male_mask, "survived"].mean(),
            ],
        }
    )

    class_rows = []
    sex_class_rows = []
    for passenger_class in sorted(clean_df["pclass"].unique()):
        class_mask = clean_df["pclass"] == passenger_class
        class_rows.append(
            {
                "pclass": int(passenger_class),
                "survival_rate": clean_df.loc[class_mask, "survived"].mean(),
            }
        )
        for sex in ["female", "male"]:
            combined_mask = (clean_df["pclass"] == passenger_class) & (clean_df["sex"] == sex)
            sex_class_rows.append(
                {
                    "sex": sex,
                    "pclass": int(passenger_class),
                    "survival_rate": clean_df.loc[combined_mask, "survived"].mean(),
                }
            )
    survival_by_class = pd.DataFrame(class_rows)
    survival_by_sex_class = pd.DataFrame(sex_class_rows)
    survival_by_sex.to_csv(OUTPUT_DIR / "02_survival_by_sex.csv", index=False)
    survival_by_class.to_csv(OUTPUT_DIR / "03_survival_by_class.csv", index=False)
    survival_by_sex_class.to_csv(OUTPUT_DIR / "04_survival_by_sex_class.csv", index=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=survival_by_sex, x="sex", y="survival_rate", hue="sex", legend=False)
    plt.ylim(0, 1)
    plt.title("Survival rate by sex")
    save_chart("02_survival_by_sex.png")

    plt.figure(figsize=(7, 5))
    sns.barplot(data=survival_by_class, x="pclass", y="survival_rate", hue="pclass", legend=False)
    plt.ylim(0, 1)
    plt.title("Survival rate by passenger class")
    save_chart("03_survival_by_class.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=survival_by_sex_class, x="pclass", y="survival_rate", hue="sex")
    plt.ylim(0, 1)
    plt.title("Survival rate by sex and passenger class")
    save_chart("04_survival_by_sex_class.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=clean_df, x="survived", y="age", hue="survived", legend=False)
    plt.xticks([0, 1], ["Did not survive", "Survived"])
    plt.title("Age distribution by survival outcome")
    save_chart("05_age_by_survival.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=clean_df, x="survived", y="fare", hue="survived", legend=False)
    plt.xticks([0, 1], ["Did not survive", "Survived"])
    plt.title("Fare distribution by survival outcome")
    save_chart("06_fare_by_survival.png")

    correlation_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    correlation = clean_df[correlation_columns].corr()
    correlation.to_csv(OUTPUT_DIR / "05_correlation_matrix.csv")
    strongest = strongest_correlations(correlation)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    save_chart("07_correlation_heatmap.png")

    standardized = clean_df[["age", "fare"]].copy()
    before = standardized.agg(["mean", "std"])
    standardized = (standardized - standardized.mean()) / standardized.std()
    after = standardized.agg(["mean", "std"])
    standardization = pd.concat({"before": before, "after": after})
    standardization.to_csv(OUTPUT_DIR / "06_standardization_check.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(standardized["age"], kde=True, ax=axes[0])
    axes[0].set_title("Standardized age")
    sns.histplot(standardized["fare"], kde=True, ax=axes[1])
    axes[1].set_title("Standardized fare")
    save_chart("08_standardized_age_fare.png")

    summary = {
        "raw_shape": list(df.shape),
        "clean_shape": list(clean_df.shape),
        "missing_percentages": {key: round(float(value), 2) for key, value in missing_percentages.items()},
        "age_outliers": age_outliers,
        "age_iqr_bounds": [round(age_lower, 2), round(age_upper, 2)],
        "fare_outliers": fare_outliers,
        "fare_iqr_bounds": [round(fare_lower, 2), round(fare_upper, 2)],
        "fare_mean": round(fare_mean, 2),
        "fare_median": round(fare_median, 2),
        "fare_mode": round(fare_mode, 2),
        "strongest_correlations": strongest,
        "class_balance": {
            str(key): round(float(value), 4)
            for key, value in clean_df["survived"].value_counts(normalize=True).sort_index().items()
        },
    }
    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved raw dataset with {len(df)} rows.")
    print(f"Saved cleaned dataset with {len(clean_df)} rows.")
    print("EDA completed successfully.")


if __name__ == "__main__":
    main()
