# Pavan_CapStone_MasaiAI
# Zepto Data and AI Platform

repository containing the three connected capstone modules.

## Modules
# Module 1 – Data Pipeline

## Overview

This module creates a complete data pipeline using book information from [Books to Scrape](https://books.toscrape.com/). The website is designed for practising web scraping.

The pipeline performs the following steps:

1. Scrapes book information from three categories.
2. Cleans and converts the scraped values into suitable data types.
3. Converts book prices from GBP to INR.
4. Stores the cleaned data in a normalized SQLite database.
5. Executes SQL queries against the database.
6. Compares an SQL join result with the same operation performed using `pandas.merge()`.

## Data Collected

The following categories were selected:

-- Travel
-- Mystery
-- Historical Fiction

The final dataset contains 69 books across these three categories.

For each book, the following information was collected:

-- Title
-- Price in GBP
-- Star rating
-- Availability
-- Category

The script uses the `requests` library to download the pages and `BeautifulSoup` to extract the required information from the HTML.

## Data Cleaning

The scraped values were cleaned using the following steps:

-- The GBP currency symbol was removed from the price.
-- The price was converted to a floating-point value and stored as `price_gbp`.
-- Text ratings such as `One`, `Two`, and `Five` were converted to integers from 1 to 5.
-- Availability text was converted into a boolean `in_stock` column.
-- Leading and trailing spaces were removed from text fields.
-- Book titles and categories were checked before loading the records into the database.

If a numeric value cannot be parsed, the script replaces it with the median value of that column. A row is removed only when its title, category, or availability cannot be interpreted correctly. This prevents one unexpected value from stopping the complete pipeline.

The cleaning actions are recorded in:

```text
output/cleaning_log.txt
```

## Currency Conversion

Book prices are converted from GBP to INR using the fixed conversion rate provided in the project instructions:

```text
1 GBP = 105.50 INR
```

The conversion is calculated as:

```python
price_inr = price_gbp -- 105.50
```

The final INR value is rounded to two decimal places.

This is a fixed project-defined rate. No external currency API is used.

## Database Design

The cleaned data is stored in a SQLite database named `books.db`.

The database contains two normalized tables.

### Categories table

```text
categories
----------
category_id
category_name
```

`category_id` is the primary key, and `category_name` is unique.

### Books table

```text
books
-----
book_id
title
price_gbp
price_inr
rating
in_stock
category_id
```

`book_id` is the primary key.

`category_id` is a foreign key that connects each book to the corresponding record in the `categories` table.

Keeping categories in a separate table avoids storing the same category name repeatedly and maintains a normalized database structure.

## SQL Queries

Six SQL queries are included in the pipeline.

They demonstrate the following SQL operations:

-- `SELECT`
-- `WHERE`
-- `ORDER BY`
-- `LIMIT`
-- `DISTINCT`
-- `BETWEEN`
-- `IN`
-- `JOIN`

The queries include:

1. Finding available five-star books.
2. Listing the ten most expensive books.
3. Displaying the distinct rating values.
4. Finding books priced between £20 and £40.
5. Finding books with ratings of four or five.
6. Joining the `books` and `categories` tables.

Each query result is saved as a CSV file in the `output` directory. The complete SQL statements and their readable outputs are also available in:

```text
output/query_outputs.txt
```

## Pandas and SQL Comparison

Query results are read from SQLite using `pd.read_sql()`.

The join between the `books` and `categories` tables is also recreated using `pd.merge()` on the pandas DataFrames.

The two results are sorted into the same order and compared using:

```python
pd.testing.assert_frame_equal()
```

The comparison completed successfully, confirming that the SQL join and pandas merge produced equivalent results.

The comparison result is saved in:

```text
output/09_join_comparison.txt
```

## Project Structure

```text
data_pipeline/
├── pipeline.py
├── requirements.txt
├── README.md
├── books.db
└── output/
    ├── books_raw.csv
    ├── books_clean.csv
    ├── cleaning_log.txt
    ├── query_outputs.txt
    ├── 01_in_stock_five_star.csv
    ├── 02_top_10_expensive.csv
    ├── 03_distinct_ratings.csv
    ├── 04_mid_price_books.csv
    ├── 05_selected_ratings.csv
    ├── 06_join_books_categories.csv
    ├── 07_category_summary_read_sql.csv
    ├── 08_join_reproduced_with_pd_merge.csv
    └── 09_join_comparison.txt
```

## Installation

Open a terminal in the project repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

For macOS or Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r data_pipeline/requirements.txt
```

## Running the Pipeline

Run the following command from the repository root:

```bash
python data_pipeline/pipeline.py
```

The script will:

-- Scrape the selected book categories.
-- Create the raw CSV file.
-- Clean and validate the data.
-- Create the converted INR price.
-- Recreate the SQLite database.
-- Execute all SQL queries.
-- Save the query results.
-- Compare the SQL join with the pandas merge.

A successful run displays output similar to:

```text
Success: 69 books across 3 categories.
Database: data_pipeline/books.db
PASS: SQL JOIN and pandas merge outputs match.
```

## Output Files

`books_raw.csv` contains the values exactly as they were collected from the website.

`books_clean.csv` contains the cleaned and converted values used to create the database.

`books.db` contains the normalized SQLite tables.

The numbered CSV files contain the results of the SQL queries and pandas operations.

## Key Learning

This module demonstrates how raw website data can be converted into structured and useful information. It covers web scraping, data cleaning, type conversion, currency conversion, relational database design, SQL queries, and pandas operations in one end-to-end pipeline.


# Module 2 – Titanic Analytics Pipeline

## Overview

This module performs exploratory data analysis, data cleaning, classification, imbalance handling, hyperparameter tuning and regression using the Titanic dataset.

The module is divided into two connected stages:

1. `01_eda.py` loads the Titanic dataset, saves an offline copy, cleans the data and generates the exploratory analysis.
2. `02_modeling.py` reads the cleaned dataset, trains and evaluates the models, performs hyperparameter tuning and saves the best complete prediction pipeline.

The Titanic dataset is loaded only once using:

```python
sns.load_dataset("titanic")
```

The original dataset is immediately saved as `titanic.csv`. This allows the project to run without downloading the dataset again.

## Project Structure

```text
analytics/
├── 01_eda.py
├── 02_modeling.py
├── run_all.py
├── predict_example.py
├── requirements.txt
├── README.md
├── titanic.csv
├── titanic_clean.csv
├── best_pipeline.joblib
└── output/
    ├── charts/
    ├── 01_profile.txt
    ├── 02_survival_by_sex.csv
    ├── 03_survival_by_class.csv
    ├── 04_survival_by_sex_class.csv
    ├── 05_correlation_matrix.csv
    ├── 06_standardization_check.csv
    ├── 07_classification_metrics.csv
    ├── 08_imbalance_comparison.csv
    ├── 09_regression_metrics.csv
    ├── 10_model_comparison.csv
    ├── eda_summary.json
    └── modeling_summary.json
```

## Installation

Create and activate a virtual environment from the repository root.

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r analytics/requirements.txt
```

## Running the Module

Run the complete module from the repository root:

```bash
python analytics/run_all.py
```

The scripts can also be run separately in the required order:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

## Dataset Profile

The original Titanic dataset contains:

-- 891 rows
-- 15 columns

The complete `df.info()`, `df.describe()` and shape results are stored in:

```text
output/01_profile.txt
```

After missing-value handling, the cleaned dataset contains 889 rows and 14 columns.

## Missing-Value Handling

| Column        | Missing | Treatment                                                                            |
| ------------- | ------: | ------------------------------------------------------------------------------------ |
| `age`         |  19.87% | Imputed using the median because the missing percentage is between 5% and 30%.       |
| `embarked`    |   0.22% | Rows were removed because the missing percentage is below 5%.                        |
| `embark_town` |   0.22% | The same two affected rows were removed.                                             |
| `deck`        |  77.22% | Column was removed because too much information was missing for reliable imputation. |

Median imputation was selected for age because it is less affected by extreme values than mean imputation.

## Univariate Analysis

Histograms and box plots were created for `age` and `fare`.

Using the IQR method:

-- Age has 65 outliers outside the range 2.50 to 54.50.
-- Fare has 114 outliers outside the range -26.76 to 65.66.

The outliers were reported but not removed because they may represent genuine passengers and ticket prices.

Fare statistics:

-- Mean: 32.10
-- Median: 14.45
-- Mode: 8.05

Since mean > median > mode, fare has a right-skewed distribution. A small number of expensive tickets create the long upper tail.

## Survival Analysis

### Survival by Sex

| Sex    | Survival rate |
| ------ | ------------: |
| Female |        74.04% |
| Male   |        18.89% |

Female passengers had a substantially higher survival rate than male passengers. This indicates that sex was strongly related to the rescue outcome.

### Survival by Passenger Class

| Class  | Survival rate |
| ------ | ------------: |
| First  |        62.62% |
| Second |        47.28% |
| Third  |        24.24% |

Survival decreased from first class to third class. This suggests that class-related factors, such as cabin location and access to lifeboats, affected the outcome.

### Survival by Sex and Class

| Passenger group     | Survival rate |
| ------------------- | ------------: |
| First-class female  |        96.74% |
| First-class male    |        36.89% |
| Second-class female |        92.11% |
| Second-class male   |        15.74% |
| Third-class female  |        50.00% |
| Third-class male    |        13.54% |

Sex and class together provide a clearer picture of survival. Women in first and second class had survival rates above 92%, while third-class men had the lowest survival rate.

## Multivariate Charts

The following charts are saved in `output/charts`:

1. Survival by sex
2. Survival by passenger class
3. Survival by sex and class
4. Age distribution by survival
5. Fare distribution by survival
6. Correlation heatmap

The age distributions of survivors and non-survivors overlap considerably. This means age alone does not clearly separate the outcomes.

Survivors generally paid higher fares. Since fare is connected to passenger class, this supports the finding that higher-class passengers had better survival outcomes.

## Correlation Analysis

The correlation matrix uses exactly these columns:

```text
survived, pclass, age, sibsp, parch, fare
```

The derived boolean columns `adult_male` and `alone` were excluded.

The two strongest absolute correlations are:

1. `pclass` and `fare`: -0.548
   Lower class numbers represent better passenger classes. The negative correlation shows that first-class passengers generally paid higher fares.

2. `sibsp` and `parch`: 0.415
   Passengers travelling with spouses or siblings were also more likely to travel with parents or children.

## Standardization Check

Before standardization:

| Column |  Mean | Standard deviation |
| ------ | ----: | -----------------: |
| Age    | 29.32 |              12.98 |
| Fare   | 32.10 |              49.70 |

After applying the z-score formula, both columns have a mean approximately equal to 0 and a standard deviation equal to 1.

This standardization was performed only as an exploratory check. The modeling pipeline performs its own scaling using training data only.

## Train and Test Split

The target distribution is:

-- Did not survive: 549 passengers or 61.75%
-- Survived: 340 passengers or 38.25%

An 80/20 stratified split was used:

-- Training rows: 711
-- Test rows: 178

Stratification preserves approximately the same class distribution in the training and test datasets.

The split was completed before preprocessing to prevent data leakage.

## Preprocessing

The modeling pipeline performs the following steps:

### Numeric columns

-- Median imputation
-- Standard scaling

### Categorical columns

-- Most-frequent imputation
-- One-hot encoding

The preprocessing steps are contained inside a `ColumnTransformer` and scikit-learn `Pipeline`. They are fitted only on the training data and applied to the test data without refitting.

## Classification Results

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8090 |    0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |   0.7640 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest       |   0.8090 |    0.7656 | 0.7206 | 0.7424 | 0.8196 |

Random Forest achieved the highest F1 score and recall among the three main classifiers. Logistic Regression achieved the highest AUC.

Confusion matrices and ROC curves are stored in the charts directory. The Decision Tree is also saved with its feature and class labels.

## Imbalance Handling

Logistic Regression was evaluated using three imbalance approaches.

| Method                 | Precision | Recall |     F1 |
| ---------------------- | --------: | -----: | -----: |
| Baseline               |    0.7833 | 0.6912 | 0.7344 |
| Balanced class weights |    0.7183 | 0.7500 | 0.7338 |
| SMOTE                  |    0.7353 | 0.7353 | 0.7353 |

SMOTE was applied only to the training data after the train/test split. It produced the highest F1 score, although the improvement over the baseline was small.

Balanced class weights produced the highest recall but reduced precision. This demonstrates the trade-off between detecting more survivors and producing additional false-positive predictions.

## Random Forest Hyperparameter Tuning

`GridSearchCV` evaluated:

-- `n_estimators`
-- `max_depth`
-- `max_features`

The best parameter combination was:

```text
n_estimators = 200
max_depth = 4
max_features = sqrt
```

Results:

-- Best cross-validation F1: 0.7448
-- OOB score: 0.8172
-- Test accuracy: 0.8202
-- Test precision: 0.8333
-- Test recall: 0.6618
-- Test F1: 0.7377
-- Test AUC: 0.8376

The Random Forest estimator was created with `oob_score=True`, allowing the out-of-bag score to be reported.

## Fare Regression

A multivariate Linear Regression model was used to predict fare.

Features included:

-- Survival outcome
-- Passenger class
-- Age
-- Number of siblings or spouses
-- Number of parents or children
-- Sex
-- Embarkation port

| Metric      |  Result |
| ----------- | ------: |
| MAE         | 21.0986 |
| RMSE        | 41.7021 |
| R²          |  0.3482 |
| Adjusted R² |  0.3091 |

The residual plot shows that the spread of prediction errors increases for higher predicted fares. The correlation between absolute residual size and predicted fare is 0.5174, indicating heteroscedasticity.

The model is less consistent when predicting high fares and does not capture all the non-linear pricing patterns.

## Model Recommendation

Random Forest was selected as the final classification model because it achieved the highest main-model F1 score of 0.7424 and recall of 0.7206 while maintaining an accuracy of 0.8090.

Logistic Regression had the highest AUC of 0.8610 and would be a reasonable alternative when model interpretability is more important. Decision Tree had the lowest recall and F1 score, making it the weakest option for this dataset.

Classification and regression metrics are presented as separate groups because they measure different prediction tasks and should not be compared on a shared scale.

## Saved Pipeline

The selected complete pipeline is saved as:

```text
best_pipeline.joblib
```

The file contains:

-- Numeric imputer
-- Standard scaler
-- Categorical imputer
-- One-hot encoder
-- Random Forest classifier

It can accept raw passenger data without requiring manual scaling or encoding.

Run the reload example with:

```bash
python analytics/predict_example.py
```

The reload test confirms that the saved pipeline produces the same predictions after loading it from disk.

## Main Outputs

-- `titanic.csv` – original offline dataset
-- `titanic_clean.csv` – cleaned dataset
-- `best_pipeline.joblib` – complete fitted prediction pipeline
-- `output/01_profile.txt` – dataset profile
-- `output/07_classification_metrics.csv` – classifier comparison
-- `output/08_imbalance_comparison.csv` – imbalance results
-- `output/09_regression_metrics.csv` – regression results
-- `output/10_model_comparison.csv` – final grouped comparison
-- `output/charts/` – all analysis and evaluation charts


Module 3:



- `analytics/` — upcoming analytics and modeling pipeline.
- `support_assistant/` — upcoming GenAI support assistant.

Each module uses its own `requirements.txt`. See the module README for exact setup and run instructions.