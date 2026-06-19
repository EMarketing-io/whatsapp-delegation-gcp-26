CREATE TABLE IF NOT EXISTS config (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255),       -- employee full name (col A)
    email       VARCHAR(255),       -- employee email (col B)
    role        VARCHAR(255),       -- role / designation (col C)
    customer    VARCHAR(255),       -- client / customer name (col D)
    department  VARCHAR(255)        -- department name (col E)
);

CREATE INDEX IF NOT EXISTS idx_config_name ON config (name);
CREATE INDEX IF NOT EXISTS idx_config_email ON config (email);
CREATE INDEX IF NOT EXISTS idx_config_customer ON config (customer);
CREATE INDEX IF NOT EXISTS idx_config_department ON config (department);
