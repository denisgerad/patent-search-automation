-- storage/schema.sql
-- Reference DDL for the patents table.
-- SQLAlchemy Core creates this automatically via create_tables(); this file
-- is provided for documentation and manual inspection only.

CREATE TABLE IF NOT EXISTS patents (
    patent_id       VARCHAR(32)  PRIMARY KEY,
    patent_title    TEXT         NOT NULL,
    patent_abstract TEXT,
    patent_type     VARCHAR(64),
    patent_date     VARCHAR(16),          -- ISO date string e.g. '2023-05-09'
    created_at      DATETIME     NOT NULL DEFAULT (datetime('now'))
);
