CREATE TABLE IF NOT EXISTS tasks (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    timestamp           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    task_id             VARCHAR(20) NOT NULL UNIQUE,        -- e.g. TASK-0001
    task_description    TEXT,
    assigned_by         VARCHAR(255),                       -- sender phone/name
    assignee_contact    VARCHAR(255),                       -- sender contact info
    assigned_to         VARCHAR(255),                       -- person task is for
    employee_email_id   VARCHAR(255),                       -- assignee email
    target_date         DATE,
    priority            VARCHAR(20) DEFAULT 'Medium',       -- Low / Medium / High / Critical
    approval_needed     VARCHAR(10),                        -- Yes / No
    client_name         VARCHAR(255),
    department          VARCHAR(255),
    assigned_name       VARCHAR(255),                       -- delegator name
    assigned_email_id   VARCHAR(255),                       -- delegator email
    comments            TEXT,
    source_link         TEXT,
    status              VARCHAR(50) DEFAULT 'Pending',      -- Pending / In Progress / Done
    message_type        VARCHAR(20),                        -- text / audio / image / etc.
    updated_timestamp   DATETIME,

    INDEX idx_task_id (task_id),
    INDEX idx_status (status),
    INDEX idx_assigned_to (assigned_to),
    INDEX idx_target_date (target_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
