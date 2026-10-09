-- storage/schema.sql
-- DDL for the patents table in patent_db (MySQL).
-- Column names match PatentRecord fields exactly — no mapping required.
-- Created automatically by storage.database.create_tables() on first run.

CREATE TABLE IF NOT EXISTS patents (
    patent_id       VARCHAR(64)  NOT NULL,
    patent_title    TEXT         NOT NULL,
    patent_abstract TEXT,
    patent_type     VARCHAR(64),
    patent_date     VARCHAR(16),          -- ISO date string e.g. '2023-05-09'
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (patent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
