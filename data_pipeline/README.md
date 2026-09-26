# Data Pipeline

The default run scrapes Books to Scrape across the Travel, Mystery, Historical Fiction, and Classics categories. It follows each category's pagination, so it gathers the full listings rather than stopping after the first page.

## Run

From the repository root:

```powershell
python -m pip install -r data_pipeline/requirements.txt
python data_pipeline/pipeline.py
```

`python data_pipeline/pipeline.py --offline` is a network-free demonstration with synthetic rows. The script labels that source in its output; those generated rows are not represented as scraped books.

## Cleaning and storage

Prices are parsed into `price_gbp`; the text star rating is mapped to an integer from 1 to 5; availability becomes a boolean `in_stock` value. Unparseable numeric prices and ratings are median-imputed, and rows with missing titles, categories, or unparseable availability are dropped. `price_inr` uses the fixed project rate **1 GBP = 105.50 INR**, with no live currency lookup.

SQLite stores the unique category names in `categories` and references them from `books.category_id`. The five saved SQL statements cover sorting/limiting, a join, `IN`, `DISTINCT`, and `BETWEEN`. Query strings and printed outputs are in `data/query_outputs/queries.sql` and `query_outputs.txt`; CSV results are alongside them. The join is also recreated with `pandas.merge` and checked for equality against the SQL output.

The generated SQLite database and cleaned CSV live in `data/`.
