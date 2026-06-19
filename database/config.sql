CREATE TABLE IF NOT EXISTS config (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(255),
    email       VARCHAR(255),
    role        VARCHAR(255),
    customer    VARCHAR(255),
    department  VARCHAR(255),

    INDEX idx_name (name),
    INDEX idx_email (email),
    INDEX idx_customer (customer),
    INDEX idx_department (department)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
