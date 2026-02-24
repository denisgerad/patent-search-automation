-- storage/schema.sql
-- Existing MySQL schema for patent_db.patents (reflected, not recreated).
-- This file documents the real table structure for reference.
--
-- PatentRecord field → MySQL column mapping:
--   patent_id       → patent_number   (UNIQUE business key)
--   patent_title    → title
--   patent_abstract → abstract
--   patent_type     → metadata        (text field; stores type string)
--   patent_date     → publication_date

CREATE TABLE IF NOT EXISTS patents (
    id               INT          NOT NULL AUTO_INCREMENT,
    patent_number    VARCHAR(100) NOT NULL,
    title            TEXT,
    link             TEXT,
    metadata         TEXT,                    -- used to store patent_type
    publication_date DATE,
    abstract         TEXT,
    claims           LONGTEXT,
    created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_patent_number (patent_number)  -- added by pipeline at startup
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
