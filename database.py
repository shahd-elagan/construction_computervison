CREATE TABLE excavator_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    object_name TEXT,
    class_id INT,
    status TEXT,
    action_mode TEXT,
    dwell_time FLOAT
);