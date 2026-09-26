"""Scrape, clean, normalize, and query a small Books to Scrape dataset."""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "books.db"
OUTPUT_DIR = DATA_DIR / "query_outputs"
BASE_URL = "https://books.toscrape.com"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
CATEGORY_NAMES = ("Travel", "Mystery", "Historical Fiction", "Classics")


def fallback_books(count: int = 60) -> list[dict[str, Any]]:
    """Return deterministic data so the pipeline remains usable offline."""
    books = []
    for index in range(count):
        category = CATEGORY_NAMES[index % len(CATEGORY_NAMES)]
        books.append({
            "title": f"Offline Book {index + 1:03d}",
            "price_gbp": round(6.99 + (index % 17) * 1.25, 2),
            "rating": index % 5 + 1,
            "in_stock": index % 9 != 0,
            "category": category,
            "source_url": f"offline://book/{index + 1}",
        })
    return books


def scrape_books(minimum: int = 60, timeout: int = 10) -> tuple[list[dict[str, Any]], str]:
    """Scrape every listing page in four categories; never mislabel fallback data as scraped."""
    try:
        import requests
        from bs4 import BeautifulSoup

        homepage = requests.get(f"{BASE_URL}/", timeout=timeout)
        homepage.raise_for_status()
        home_soup = BeautifulSoup(homepage.text, "html.parser")
        wanted = {"Travel", "Mystery", "Historical Fiction", "Classics"}
        category_links = [
            (anchor.get_text(" ", strip=True), urljoin(f"{BASE_URL}/", anchor.get("href", "")))
            for anchor in home_soup.select("aside.sidebar ul.nav-list li ul li a")
            if anchor.get_text(" ", strip=True) in wanted and anchor.get("href")
        ]
        if len(category_links) < 3:
            raise RuntimeError("Could not discover at least three requested category links from the homepage.")

        books: list[dict[str, Any]] = []
        for category_name, category_url in category_links:
            current_url = category_url
            while current_url:
                response = requests.get(current_url, timeout=timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for card in soup.select("article.product_pod"):
                    title_node = card.select_one("h3 a")
                    price_node = card.select_one(".price_color")
                    rating_node = card.select_one(".star-rating")
                    stock_node = card.select_one(".availability")
                    if not all((title_node, price_node, rating_node, stock_node)):
                        continue
                    rating_classes = rating_node.get("class", [])
                    star_rating = next((value for value in rating_classes if value in RATING_MAP), "")
                    books.append({
                        "title": title_node.get("title") or title_node.get_text(strip=True),
                        "price": price_node.get_text(strip=True),
                        "star_rating": star_rating,
                        "availability": stock_node.get_text(" ", strip=True),
                        "category": category_name,
                        "source_url": urljoin(current_url, title_node.get("href", "")),
                    })
                next_page = soup.select_one("li.next a")
                current_url = urljoin(current_url, next_page["href"]) if next_page else ""
        if len(books) >= minimum:
            return books, "web"
        raise RuntimeError(f"Scrape found only {len(books)} books; at least {minimum} are required.")
    except ImportError:
        raise RuntimeError("Install data_pipeline/requirements.txt before running the scraper.") from None
    except requests.RequestException as exc:
        raise RuntimeError(f"Books to Scrape could not be reached: {exc}. Use --offline only for a demo run.") from exc


def clean_books(raw_books: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse fields; median-impute numeric parse failures and drop invalid stock text."""
    frame = pd.DataFrame(raw_books).copy()
    if "price" not in frame and "price_gbp" in frame:
        frame["price"] = frame["price_gbp"]
    if "star_rating" not in frame and "rating" in frame:
        frame["star_rating"] = frame["rating"]
    if "availability" not in frame and "in_stock" in frame:
        frame["availability"] = frame["in_stock"].map(lambda value: "In stock" if bool(value) else "Out of stock")

    frame["title"] = frame.get("title", pd.Series(dtype=str)).astype("string").str.strip()
    frame["category"] = frame.get("category", pd.Series(dtype=str)).astype("string").str.strip()
    frame["price_gbp"] = pd.to_numeric(
        frame["price"].astype("string").str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )
    frame["rating"] = pd.to_numeric(
        frame["star_rating"].map(lambda value: RATING_MAP.get(str(value), value)),
        errors="coerce",
    )
    numeric_bad = frame["price_gbp"].isna() | frame["rating"].isna()
    dropped_numeric_rows = int(numeric_bad.sum())
    for column in ("price_gbp", "rating"):
        frame[column] = frame[column].fillna(frame[column].median())
    frame["rating"] = frame["rating"].round().clip(1, 5).astype(int)

    availability = frame["availability"].astype("string").str.lower()
    frame["in_stock"] = availability.map(
        lambda text: True if isinstance(text, str) and "in stock" in text
        else False if isinstance(text, str) and "out of stock" in text
        else pd.NA
    )
    invalid_stock = frame["in_stock"].isna() | frame["title"].isna() | frame["category"].isna()
    invalid_stock |= frame["title"].eq("") | frame["category"].eq("")
    dropped_invalid_rows = int(invalid_stock.sum())
    frame = frame.loc[~invalid_stock].drop_duplicates(subset=["title", "category"]).copy()
    frame["in_stock"] = frame["in_stock"].astype(bool)
    frame["price_inr"] = (frame["price_gbp"] * 105.50).round(2)
    frame["source_url"] = frame.get("source_url", "")
    frame.attrs["cleaning_decisions"] = {
        "numeric_parse_failures_imputed_with_median": dropped_numeric_rows,
        "rows_dropped_for_missing_title_category_or_unparseable_availability": dropped_invalid_rows,
        "inr_conversion_rate_gbp": 105.50,
    }
    return frame[["title", "price_gbp", "price_inr", "rating", "in_stock", "category", "source_url"]]


def load_database(frame: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript("""
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL REFERENCES categories(category_id),
                source_url TEXT NOT NULL
            );
        """)
        categories = sorted(frame["category"].unique())
        connection.executemany("INSERT INTO categories(name) VALUES (?)", [(name,) for name in categories])
        category_ids = {name: index + 1 for index, name in enumerate(categories)}
        rows = [(
            row.title, row.price_gbp, row.price_inr, row.rating, int(row.in_stock),
            category_ids[row.category], row.source_url,
        ) for row in frame.itertuples(index=False)]
        connection.executemany(
            """INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id, source_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""", rows,
        )


def run_queries(
    frame: pd.DataFrame,
    db_path: Path = DB_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Execute the assignment SQL and compare its JOIN to an in-memory pandas merge."""
    queries = {
        "01_books_by_price": "SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10",
        "02_books_join_categories": "SELECT b.title, c.name AS category, b.price_gbp, b.rating, b.in_stock FROM books AS b JOIN categories AS c ON b.category_id = c.category_id ORDER BY c.name, b.title",
        "03_high_rated": "SELECT title, rating, price_gbp FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, price_gbp DESC",
        "04_distinct_categories": "SELECT DISTINCT name AS category FROM categories ORDER BY category",
        "05_price_range": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp DESC LIMIT 10",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    (output_dir / "queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{query};" for name, query in queries.items()) + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as connection:
        for name, query in queries.items():
            result = pd.read_sql(query, connection)
            if name == "02_books_join_categories":
                result["in_stock"] = result["in_stock"].astype(bool)
            result.to_csv(output_dir / f"{name}.csv", index=False)
            results[name] = result
        pd.read_sql(queries["01_books_by_price"], connection)

    categories = pd.DataFrame({"name": sorted(frame["category"].unique())})
    categories["category_id"] = range(1, len(categories) + 1)
    merged = frame.merge(categories, left_on="category", right_on="name", how="inner")
    merged = merged[["title", "category", "price_gbp", "rating", "in_stock"]]
    merged = merged.sort_values(["category", "title"], kind="stable").reset_index(drop=True)
    merged["title"] = merged["title"].astype(str)
    merged["category"] = merged["category"].astype(str)
    merged["price_gbp"] = merged["price_gbp"].astype(float)
    merged["in_stock"] = merged["in_stock"].astype(bool)
    sql_join = results["02_books_join_categories"].reset_index(drop=True)
    if not sql_join.equals(merged):
        raise AssertionError("SQL JOIN output and in-memory pandas merge differ.")
    merged.to_csv(output_dir / "06_pandas_merge.csv", index=False)
    results["06_pandas_merge"] = merged
    with (output_dir / "query_outputs.txt").open("w", encoding="utf-8") as output:
        for name, result in results.items():
            output.write(f"{name}\n{result.to_string(index=False)}\n\n")
    return results


def main() -> None:
    # Book titles can include characters outside the default Windows code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use generated data without making a request.")
    args = parser.parse_args()
    if args.offline:
        raw, source = fallback_books(), "synthetic offline demonstration data"
    else:
        raw, source = scrape_books()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw).to_csv(DATA_DIR / "books_scraped_raw.csv", index=False)
    frame = clean_books(raw)
    frame.to_csv(DATA_DIR / "books_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    load_database(frame)
    results = run_queries(frame)
    print(f"Loaded {len(frame)} books from {source} across {frame['category'].nunique()} categories.")
    for name, result in results.items():
        print(f"\n{name} ({len(result)} rows):\n{result.to_string(index=False)}")


if __name__ == "__main__":
    main()
