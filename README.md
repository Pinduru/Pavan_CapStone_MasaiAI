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

* Travel
* Mystery
* Historical Fiction

The final dataset contains 69 books across these three categories.

For each book, the following information was collected:

* Title
* Price in GBP
* Star rating
* Availability
* Category

The script uses the `requests` library to download the pages and `BeautifulSoup` to extract the required information from the HTML.

## Data Cleaning

The scraped values were cleaned using the following steps:

* The GBP currency symbol was removed from the price.
* The price was converted to a floating-point value and stored as `price_gbp`.
* Text ratings such as `One`, `Two`, and `Five` were converted to integers from 1 to 5.
* Availability text was converted into a boolean `in_stock` column.
* Leading and trailing spaces were removed from text fields.
* Book titles and categories were checked before loading the records into the database.

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
price_inr = price_gbp * 105.50
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

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `BETWEEN`
* `IN`
* `JOIN`

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

* Scrape the selected book categories.
* Create the raw CSV file.
* Clean and validate the data.
* Create the converted INR price.
* Recreate the SQLite database.
* Execute all SQL queries.
* Save the query results.
* Compare the SQL join with the pandas merge.

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



Module 2:

Module 3:



- `analytics/` — upcoming analytics and modeling pipeline.
- `support_assistant/` — upcoming GenAI support assistant.

Each module uses its own `requirements.txt`. See the module README for exact setup and run instructions.