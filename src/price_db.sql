CREATE TABLE IF NOT EXISTS prices (
    id INTEGER,
    ts INTEGER,
    high INTEGER,
    hvol INTEGER,
    low INTEGER,
    lvol INTEGER,
    PRIMARY KEY (id, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS item_map (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    examine TEXT,
    lmt INTEGER,
    value INTEGER,
    members INT
) WITHOUT ROWID;