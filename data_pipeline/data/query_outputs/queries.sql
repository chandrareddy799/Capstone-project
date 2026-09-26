-- 01_books_by_price
SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 10;

-- 02_books_join_categories
SELECT b.title, c.name AS category, b.price_gbp, b.rating, b.in_stock FROM books AS b JOIN categories AS c ON b.category_id = c.category_id ORDER BY c.name, b.title;

-- 03_high_rated
SELECT title, rating, price_gbp FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, price_gbp DESC;

-- 04_distinct_categories
SELECT DISTINCT name AS category FROM categories ORDER BY category;

-- 05_price_range
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp DESC LIMIT 10;
