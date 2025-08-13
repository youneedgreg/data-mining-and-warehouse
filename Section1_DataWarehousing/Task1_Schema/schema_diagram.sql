-- Star Schema Design for Retail Data Warehouse
-- Task 1: Data Warehouse Design (15 Marks)

-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS SalesFact;
DROP TABLE IF EXISTS CustomerDim;
DROP TABLE IF EXISTS ProductDim;
DROP TABLE IF EXISTS TimeDim;
DROP TABLE IF EXISTS StoreDim;

-- 1. Time Dimension Table
CREATE TABLE TimeDim (
    time_id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    day INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    year INTEGER NOT NULL,
    day_of_week VARCHAR(10),
    month_name VARCHAR(10),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);

-- 2. Customer Dimension Table
CREATE TABLE CustomerDim (
    customer_id INTEGER PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    country VARCHAR(50) NOT NULL,
    registration_date DATE,
    customer_segment VARCHAR(20),
    age_group VARCHAR(20)
);

-- 3. Product Dimension Table
CREATE TABLE ProductDim (
    product_id INTEGER PRIMARY KEY,
    product_code VARCHAR(20) NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    brand VARCHAR(50),
    unit_price DECIMAL(10, 2),
    unit_cost DECIMAL(10, 2),
    supplier VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);

-- 4. Store Dimension Table
CREATE TABLE StoreDim (
    store_id INTEGER PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    store_type VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    country VARCHAR(50),
    region VARCHAR(50),
    store_size VARCHAR(20),
    opened_date DATE,
    manager_name VARCHAR(100)
);

-- 5. Sales Fact Table (Central table in star schema)
CREATE TABLE SalesFact (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    invoice_no VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    tax_amount DECIMAL(10, 2) DEFAULT 0,
    total_amount DECIMAL(10, 2) NOT NULL,
    profit_amount DECIMAL(10, 2),
    payment_method VARCHAR(20),
    
    -- Foreign Key Constraints
    FOREIGN KEY (time_id) REFERENCES TimeDim(time_id),
    FOREIGN KEY (customer_id) REFERENCES CustomerDim(customer_id),
    FOREIGN KEY (product_id) REFERENCES ProductDim(product_id),
    FOREIGN KEY (store_id) REFERENCES StoreDim(store_id)
);

-- Create indexes for better query performance
CREATE INDEX idx_sales_time ON SalesFact(time_id);
CREATE INDEX idx_sales_customer ON SalesFact(customer_id);
CREATE INDEX idx_sales_product ON SalesFact(product_id);
CREATE INDEX idx_sales_store ON SalesFact(store_id);
CREATE INDEX idx_time_date ON TimeDim(date);
CREATE INDEX idx_product_category ON ProductDim(category);
CREATE INDEX idx_customer_country ON CustomerDim(country);

-- Example Views for common OLAP operations

-- View for Sales by Quarter and Category
CREATE VIEW SalesByQuarterCategory AS
SELECT 
    t.year,
    t.quarter,
    p.category,
    SUM(s.total_amount) as total_sales,
    SUM(s.quantity) as total_quantity,
    COUNT(DISTINCT s.customer_id) as unique_customers
FROM SalesFact s
JOIN TimeDim t ON s.time_id = t.time_id
JOIN ProductDim p ON s.product_id = p.product_id
GROUP BY t.year, t.quarter, p.category;

-- View for Customer Demographics Analysis
CREATE VIEW CustomerDemographicsAnalysis AS
SELECT 
    c.country,
    c.customer_segment,
    c.age_group,
    COUNT(DISTINCT s.customer_id) as customer_count,
    SUM(s.total_amount) as total_revenue,
    AVG(s.total_amount) as avg_order_value
FROM SalesFact s
JOIN CustomerDim c ON s.customer_id = c.customer_id
GROUP BY c.country, c.customer_segment, c.age_group;