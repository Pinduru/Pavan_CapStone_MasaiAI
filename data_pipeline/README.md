# Module 1 — Data Pipeline

This module scrapes public book data, cleans and enriches it, loads it into a normalized SQLite database, runs the required SQL queries, and verifies that a SQL join and a pandas merge return equivalent data.

## Scope and design

- Source: `https://books.toscrape.com/`, a scraping-practice website.
- Categories: **Travel, Mystery, and Historical Fiction**. Together they provide at least 60 books.
- Currency rule: the required fixed project baseline **1 GBP = 105.50 INR**. No currency API is used.
- Schema: `categories` stores each category once; `books.category_id` is a foreign key to `categories.category_id`.
- Reliability: HTTP errors raise a clear exception, requests use a 30-second timeout, and a short delay is included between requests.

## Install and run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r data_pipeline/requirements.txt
python data_pipeline/pipeline.py
```

The command recreates `books.db` and everything inside `output/`. It prints the row/category counts and the join-comparison result.

## Cleaning decisions

- The price's currency symbol is removed with a regular expression and the result becomes `float64` in `price_gbp`.
- Text ratings `One` through `Five` become integers 1 through 5.
- Availability becomes a boolean `in_stock` value.
- `price_inr` is `(price_gbp * 105.50)`, rounded to two decimal places.
- An unparseable numeric price or rating is median-imputed so one malformed product cannot crash the pipeline. A row is dropped only when its title/category is unusable or its availability cannot be recognized, because inventing text or stock status would be misleading. The actual actions taken are recorded in `output/cleaning_log.txt`.

## SQL coverage and outputs

`pipeline.py` keeps all SQL strings in the `QUERIES` dictionary. The six queries collectively demonstrate:

1. `SELECT` and `WHERE`
2. `ORDER BY` and `LIMIT`
3. `DISTINCT`
4. `BETWEEN`
5. `IN`
6. An inner `JOIN` between `books` and `categories`

Each result is saved as CSV and all readable query text/output is saved in `output/query_outputs.txt`. At least two results are loaded with `pd.read_sql()`. The full join is independently reproduced with `pd.merge()`; `pd.testing.assert_frame_equal()` proves equivalence and writes the result to `output/09_join_comparison.txt`.

## Generated artifacts

- `books.db` — normalized SQLite database
- `output/books_raw.csv` — raw scraped fields
- `output/books_clean.csv` — typed, enriched data
- `output/01_*csv` through `output/08_...csv` — query/merge results
- `output/query_outputs.txt` — SQL strings with executed output
- `output/09_join_comparison.txt` — explicit equivalence result
