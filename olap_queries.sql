-- OLAP Queries for Retail Data Warehouse
-- Task 3: OLAP Queries and Analysis

-- 1. ROLL-UP QUERY: Total sales by country and quarter
SELECT 
    c.Country,
    t.year,
    t.quarter,
    SUM(s.total_amount) as total_sales,
    COUNT(DISTINCT s.invoice_no) as num_transactions,
    AVG(s.total_amount) as avg_transaction_value
FROM SalesFact s
JOIN CustomerDim c ON s.customer_id = c.customer_id
JOIN TimeDim t ON s.time_id = t.time_id
GROUP BY c.Country, t.year, t.quarter
ORDER BY c.Country, t.year, t.quarter;

-- 2. DRILL-DOWN QUERY: Sales details for USA by month
SELECT 
    t.year,
    t.month,
    t.month_name,
    p.category,
    SUM(s.total_amount) as total_sales,
    SUM(s.quantity) as total_quantity,
    COUNT(DISTINCT s.customer_id) as unique_customers
FROM SalesFact s
JOIN CustomerDim c ON s.customer_id = c.customer_id
JOIN TimeDim t ON s.time_id = t.time_id
JOIN ProductDim p ON s.product_id = p.product_id
WHERE c.Country = 'USA'
GROUP BY t.year, t.month, t.month_name, p.category
ORDER BY t.year, t.month;

-- 3. SLICE QUERY: Total sales for Electronics category
SELECT 
    p.product_name,
    c.Country,
    SUM(s.quantity) as total_quantity_sold,
    SUM(s.total_amount) as total_revenue,
    AVG(s.unit_price) as avg_price,
    COUNT(DISTINCT s.customer_id) as unique_customers
FROM SalesFact s
JOIN ProductDim p ON s.product_id = p.product_id
JOIN CustomerDim c ON s.customer_id = c.customer_id
WHERE p.category = 'Electronics'
GROUP BY p.product_name, c.Country
ORDER BY total_revenue DESC
LIMIT 20;
