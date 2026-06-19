CREATE TABLE IF NOT EXISTS tasks (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL DEFAULT NOW(),
    task_id             VARCHAR(20) UNIQUE NOT NULL,        -- e.g. TASK-0001
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
    updated_timestamp   TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_task_id ON tasks (task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks (assigned_to);
CREATE INDEX IF NOT EXISTS idx_tasks_target_date ON tasks (target_date);
