CREATE TABLE IF NOT EXISTS message_logs (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    timestamp   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sender      VARCHAR(255),
    msg_type    VARCHAR(50),
    raw_text    TEXT,
    task_id     VARCHAR(20),
    error       TEXT,

    INDEX idx_sender (sender),
    INDEX idx_task_id (task_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
