-- ============================================================================
-- NETFLIX DATA ANALYST PORTFOLIO: ADVANCED SQL ANALYTICAL QUERIES
-- Database Engine: SQLite 3
-- Author: Data Analyst Portfolio
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: Content Type Breakdown & Growth Percentage
-- Concept: Aggregations, ROUND, Percentage Calculation
-- ----------------------------------------------------------------------------
SELECT 
    type,
    COUNT(*) AS total_titles,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shows), 2) AS percentage_share
FROM shows
GROUP BY type;

-- ----------------------------------------------------------------------------
-- QUERY 2: Top 10 Directors with Most Content Produced
-- Concept: INNER JOIN, GROUP BY, ORDER BY, LIMIT
-- ----------------------------------------------------------------------------
SELECT 
    d.name AS director_name,
    COUNT(sd.show_id) AS total_shows_directed
FROM directors d
JOIN show_directors sd ON d.director_id = sd.director_id
GROUP BY d.director_id, d.name
ORDER BY total_shows_directed DESC
LIMIT 10;

-- ----------------------------------------------------------------------------
-- QUERY 3: Year-over-Year Content Addition Velocity & Growth Trajectory
-- Concept: CTE (Common Table Expression), Window Function (LAG)
-- ----------------------------------------------------------------------------
WITH YearlyContent AS (
    SELECT 
        CAST(SUBSTR(date_added, -4) AS INTEGER) AS year_added,
        COUNT(*) AS titles_added
    FROM shows
    WHERE date_added IS NOT NULL AND date_added != ''
    GROUP BY year_added
)
SELECT 
    year_added,
    titles_added,
    LAG(titles_added, 1, 0) OVER (ORDER BY year_added) AS previous_year_titles,
    ROUND(
        (titles_added - LAG(titles_added, 1, 0) OVER (ORDER BY year_added)) * 100.0 / 
        NULLIF(LAG(titles_added, 1, 0) OVER (ORDER BY year_added), 0), 2
    ) AS yoy_growth_percentage
FROM YearlyContent
WHERE year_added >= 2010
ORDER BY year_added DESC;

-- ----------------------------------------------------------------------------
-- QUERY 4: Top 5 Genres Per Content Type (Movies vs TV Shows)
-- Concept: CTE, Window Function (ROW_NUMBER / RANK), Multi-table Joins
-- ----------------------------------------------------------------------------
WITH RankedGenres AS (
    SELECT 
        s.type,
        g.name AS genre_name,
        COUNT(*) AS total_count,
        ROW_NUMBER() OVER (PARTITION BY s.type ORDER BY COUNT(*) DESC) AS rank_position
    FROM shows s
    JOIN show_genres sg ON s.show_id = sg.show_id
    JOIN genres g ON sg.genre_id = g.genre_id
    GROUP BY s.type, g.name
)
SELECT 
    type,
    rank_position,
    genre_name,
    total_count
FROM RankedGenres
WHERE rank_position <= 5
ORDER BY type, rank_position;

-- ----------------------------------------------------------------------------
-- QUERY 5: Top 10 Actors Featured in Most Netflix Titles
-- Concept: INNER JOIN, HAVING clause, Aggregations
-- ----------------------------------------------------------------------------
SELECT 
    a.name AS actor_name,
    COUNT(sc.show_id) AS total_titles_featured
FROM actors a
JOIN show_cast sc ON a.actor_id = sc.actor_id
GROUP BY a.actor_id, a.name
HAVING COUNT(sc.show_id) >= 15
ORDER BY total_titles_featured DESC
LIMIT 10;

-- ----------------------------------------------------------------------------
-- QUERY 6: Country Co-production Analysis (Multi-Country Collaborations)
-- Concept: Multi-join, Count of Distinct Junctions
-- ----------------------------------------------------------------------------
SELECT 
    c.name AS country_name,
    COUNT(DISTINCT s.show_id) AS total_titles,
    SUM(CASE WHEN s.type = 'Movie' THEN 1 ELSE 0 END) AS movies_count,
    SUM(CASE WHEN s.type = 'TV Show' THEN 1 ELSE 0 END) AS tv_shows_count
FROM countries c
JOIN show_countries sc ON c.country_id = sc.country_id
JOIN shows s ON sc.show_id = s.show_id
GROUP BY c.country_id, c.name
ORDER BY total_titles DESC
LIMIT 10;

-- ----------------------------------------------------------------------------
-- QUERY 7: Content Age Gap (Difference Between Release Year & Added Year)
-- Concept: Substring extraction, AVG, MAX, MIN metrics
-- ----------------------------------------------------------------------------
SELECT 
    s.type,
    ROUND(AVG(CAST(SUBSTR(s.date_added, -4) AS INTEGER) - s.release_year), 2) AS avg_years_to_platform,
    MIN(CAST(SUBSTR(s.date_added, -4) AS INTEGER) - s.release_year) AS min_gap_years,
    MAX(CAST(SUBSTR(s.date_added, -4) AS INTEGER) - s.release_year) AS max_gap_years
FROM shows s
WHERE s.date_added IS NOT NULL 
  AND s.date_added != ''
  AND CAST(SUBSTR(s.date_added, -4) AS INTEGER) >= s.release_year
GROUP BY s.type;

-- ----------------------------------------------------------------------------
-- QUERY 8: Rating Classification Distribution by Target Audience
-- Concept: CASE WHEN Conditional Classification & Percentage
-- ----------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN rating IN ('PG-13', 'TV-14') THEN 'Teens (13-14+)'
        WHEN rating IN ('TV-MA', 'R', 'NC-17') THEN 'Adults (18+)'
        WHEN rating IN ('PG', 'TV-PG', 'TV-Y7', 'TV-Y7-FV') THEN 'Older Kids (7+)'
        WHEN rating IN ('G', 'TV-Y') THEN 'Little Kids (All)'
        ELSE 'Unrated / Other'
    END AS target_audience,
    COUNT(*) AS title_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM shows), 2) AS share_pct
FROM shows
GROUP BY target_audience
ORDER BY title_count DESC;
