"""Scrape books, clean the data and save it in SQLite."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORIES = ("Travel", "Mystery", "Historical Fiction")
GBP_TO_INR = 105.50
OUT_DIR = Path(__file__).resolve().parent / "output"
DB_PATH = Path(__file__).resolve().parent / "books.db"
HEADERS = {"User-Agent": "Mozilla/5.0"}
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Download a page and return parsed HTML."""
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(0.10)
    return BeautifulSoup(response.text, "html.parser")


def category_urls(session: requests.Session) -> dict[str, str]:
    soup = get_soup(session, BASE_URL)
    links = {}
    for anchor in soup.select(".side_categories ul li ul li a"):
        name = anchor.get_text(strip=True)
        if name in CATEGORIES:
            links[name] = urljoin(BASE_URL, anchor["href"])
    missing = set(CATEGORIES) - set(links)
    if missing:
        raise RuntimeError(f"Could not find category links: {sorted(missing)}")
    return links


def scrape_books() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with requests.Session() as session:
        for category, first_url in category_urls(session).items():
            page_url: str | None = first_url
            while page_url:
                soup = get_soup(session, page_url)
                for card in soup.select("article.product_pod"):
                    title_link = card.select_one("h3 a")
                    price = card.select_one(".price_color")
                    rating = card.select_one("p.star-rating")
                    availability = card.select_one(".availability")
                    rows.append(
                        {
                            "title": title_link.get("title", "") if title_link else "",
                            "price_raw": price.get_text(strip=True) if price else None,
                            "star_rating_raw": (
                                next((c for c in rating.get("class", []) if c in RATING_MAP), None)
                                if rating else None
                            ),
                            "availability_raw": availability.get_text(" ", strip=True) if availability else None,
                            "category": category,
                        }
                    )
                next_link = soup.select_one("li.next a")
                page_url = urljoin(page_url, next_link["href"]) if next_link else None
    return pd.DataFrame(rows)


def clean_books(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert the scraped text into useful data types."""
    df = raw.copy()
    notes: list[str] = []
    df["title"] = df["title"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df["price_gbp"] = pd.to_numeric(
        df["price_raw"].astype("string").str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce",
    )
    df["rating"] = pd.to_numeric(df["star_rating_raw"].map(RATING_MAP), errors="coerce")

    for column in ("price_gbp", "rating"):
        failed = int(df[column].isna().sum())
        if failed:
            median = float(df[column].median())
            df[column] = df[column].fillna(median)
            notes.append(f"Median-imputed {failed} unparseable {column} value(s) with {median}.")

    availability = df["availability_raw"].astype("string").str.lower()
    known_availability = availability.str.contains(r"in stock|out of stock", regex=True, na=False)
    bad_text = df["title"].isna() | df["title"].eq("") | df["category"].isna() | ~known_availability
    dropped = int(bad_text.sum())
    if dropped:
        notes.append(f"Dropped {dropped} row(s) missing title/category or recognizable availability.")
        df = df.loc[~bad_text].copy()
        availability = availability.loc[~bad_text]

    df["rating"] = df["rating"].round().clip(1, 5).astype("int64")
    df["in_stock"] = availability.str.contains(r"\bin stock\b", regex=True).astype("bool")
    df["price_gbp"] = df["price_gbp"].astype("float64").round(2)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    clean = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]
    if not notes:
        notes.append("No parse failures were found; no rows required imputation or removal.")
    return clean.reset_index(drop=True), notes


def load_database(df: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL CHECK(price_gbp >= 0),
                price_inr REAL NOT NULL CHECK(price_inr >= 0),
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
            """
        )
        categories = sorted(df["category"].unique())
        connection.executemany(
            "INSERT INTO categories(category_name) VALUES (?)", [(c,) for c in categories]
        )
        category_ids = dict(connection.execute("SELECT category_name, category_id FROM categories"))
        rows = [
            (r.title, r.price_gbp, r.price_inr, int(r.rating), int(r.in_stock), category_ids[r.category])
            for r in df.itertuples(index=False)
        ]
        connection.executemany(
            """INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )


QUERIES = {
    "01_in_stock_five_star": """
        SELECT title, price_gbp, rating FROM books
        WHERE in_stock = 1 AND rating = 5 ORDER BY price_gbp DESC;
    """,
    "02_top_10_expensive": """
        SELECT title, price_gbp, price_inr FROM books
        ORDER BY price_gbp DESC LIMIT 10;
    """,
    "03_distinct_ratings": "SELECT DISTINCT rating FROM books ORDER BY rating;",
    "04_mid_price_books": """
        SELECT title, price_gbp FROM books
        WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp, title;
    """,
    "05_selected_ratings": """
        SELECT title, rating FROM books
        WHERE rating IN (4, 5) ORDER BY rating DESC, title;
    """,
    "06_join_books_categories": """
        SELECT b.title, b.price_gbp, b.price_inr, b.rating, b.in_stock, c.category_name
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.title;
    """,
}


def run_queries_and_compare(clean_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    report = ["MODULE 1 SQL QUERY OUTPUTS", "=" * 28, ""]
    with sqlite3.connect(DB_PATH) as connection:
        sql_results = {}
        for name, query in QUERIES.items():
            result = pd.read_sql(query, connection)
            sql_results[name] = result
            result.to_csv(OUT_DIR / f"{name}.csv", index=False)
            report.extend([name, query.strip(), result.to_string(index=False), ""])

        summary = pd.read_sql(
            """SELECT c.category_name, COUNT(*) AS book_count,
                      ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
               FROM books b JOIN categories c ON b.category_id = c.category_id
               GROUP BY c.category_name ORDER BY c.category_name""",
            connection,
        )
        summary.to_csv(OUT_DIR / "07_category_summary_read_sql.csv", index=False)

    categories_df = pd.DataFrame(
        {"category_id": range(1, len(sorted(clean_df.category.unique())) + 1),
         "category_name": sorted(clean_df.category.unique())}
    )
    books_df = clean_df.merge(categories_df, left_on="category", right_on="category_name")
    pandas_join = (
        books_df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_name"]]
        .sort_values(["category_name", "title"]).reset_index(drop=True)
    )
    sql_join = sql_results["06_join_books_categories"].copy()
    sql_join["in_stock"] = sql_join["in_stock"].astype(bool)
    pd.testing.assert_frame_equal(sql_join, pandas_join, check_dtype=False)
    pandas_join.to_csv(OUT_DIR / "08_join_reproduced_with_pd_merge.csv", index=False)
    comparison = "PASS: pd.read_sql JOIN and pd.merge outputs are equivalent."
    (OUT_DIR / "09_join_comparison.txt").write_text(comparison + "\n", encoding="utf-8")
    (OUT_DIR / "query_outputs.txt").write_text("\n".join(report), encoding="utf-8")


def validate(df: pd.DataFrame) -> None:
    assert len(df) >= 60, f"Expected at least 60 rows, got {len(df)}"
    assert df["category"].nunique() >= 3
    assert df["rating"].between(1, 5).all()
    assert df["in_stock"].dtype == bool
    assert ((df["price_gbp"] * GBP_TO_INR).round(2) == df["price_inr"]).all()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    raw = scrape_books()
    raw.to_csv(OUT_DIR / "books_raw.csv", index=False)
    clean, notes = clean_books(raw)
    validate(clean)
    clean.to_csv(OUT_DIR / "books_clean.csv", index=False)
    (OUT_DIR / "cleaning_log.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    load_database(clean)
    run_queries_and_compare(clean)
    print(f"Success: {len(clean)} books across {clean['category'].nunique()} categories.")
    print(f"Database: {DB_PATH}")
    print("PASS: SQL JOIN and pandas merge outputs match.")


if __name__ == "__main__":
    main()
