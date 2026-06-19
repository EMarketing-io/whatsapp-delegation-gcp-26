CREATE TABLE IF NOT EXISTS message_logs (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMP NOT NULL DEFAULT NOW(),
    sender      VARCHAR(255),       -- sender phone number with + prefix
    msg_type    VARCHAR(50),        -- text / audio / image / document / intent / command
    raw_text    TEXT,               -- original message body (capped at 500 chars)
    task_id     VARCHAR(20),        -- TASK-XXXX if this message created/updated a task
    error       TEXT                -- error message if processing failed
);

CREATE INDEX IF NOT EXISTS idx_message_logs_sender ON message_logs (sender);
CREATE INDEX IF NOT EXISTS idx_message_logs_task_id ON message_logs (task_id);
CREATE INDEX IF NOT EXISTS idx_message_logs_timestamp ON message_logs (timestamp);
