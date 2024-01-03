import sqlite3


connection = sqlite3.connect("./database.sqlite")

cursor = connection.cursor()

cursor.execute(
    """CREATE TABLE IF NOT EXISTS Experiments (
    id TEXT PRIMARY KEY,
    experiment_name TEXT,
    run_name TEXT,
    experiment_file TEXT,
    config_file TEXT,
    state TEXT NOT NULL DEFAULT 'waiting',
    created_at TEXT DEFAULT(datetime('now', 'localtime')),
    updated_at TEXT DEFAULT NULL
)"""
)

connection.commit()
