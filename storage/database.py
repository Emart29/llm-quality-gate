"""DuckDB database setup and connection management."""

import duckdb
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id              VARCHAR PRIMARY KEY,
    dataset_name    VARCHAR NOT NULL,
    dataset_version VARCHAR NOT NULL,
    provider_name   VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    commit_hash     VARCHAR,
    branch          VARCHAR,
    trigger         VARCHAR,
    total_test_cases     INTEGER NOT NULL,
    successful_executions INTEGER NOT NULL,
    failed_executions    INTEGER NOT NULL,
    total_execution_time DOUBLE NOT NULL,
    success_rate    DOUBLE NOT NULL,
    pass_rate       DOUBLE,
    quality_gate_passed BOOLEAN,
    overall_score   DOUBLE,
    task_success_score    DOUBLE,
    relevance_score       DOUBLE,
    hallucination_score   DOUBLE,
    consistency_score     DOUBLE,
    regression_detected   BOOLEAN DEFAULT FALSE,
    regression_summary    JSON,
    configuration   JSON,
    created_at      TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS test_case_results (
    id               VARCHAR PRIMARY KEY,
    run_id           VARCHAR NOT NULL REFERENCES evaluation_runs(id),
    test_case_id     VARCHAR NOT NULL,
    input_prompt     VARCHAR,
    expected_output  VARCHAR,
    generated_output VARCHAR,
    task_type        VARCHAR,
    difficulty       VARCHAR,
    execution_time   DOUBLE,
    success          BOOLEAN,
    passed           BOOLEAN,
    error            VARCHAR,
    task_success_score    DOUBLE,
    relevance_score       DOUBLE,
    hallucination_score   DOUBLE,
    consistency_score     DOUBLE,
    tags             JSON,
    metadata         JSON,
    created_at       TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS quality_gate_history (
    id              VARCHAR PRIMARY KEY,
    run_id          VARCHAR NOT NULL REFERENCES evaluation_runs(id),
    passed          BOOLEAN NOT NULL,
    overall_score   DOUBLE,
    failed_metrics  JSON,
    thresholds      JSON,
    commit_hash     VARCHAR,
    created_at      TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS provider_comparisons (
    id              VARCHAR PRIMARY KEY,
    comparison_name VARCHAR,
    dataset_name    VARCHAR NOT NULL,
    dataset_version VARCHAR NOT NULL,
    providers       JSON NOT NULL,
    results         JSON NOT NULL,
    winner          VARCHAR,
    created_at      TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS baseline_metrics (
    id              VARCHAR PRIMARY KEY,
    provider_name   VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    dataset_version VARCHAR NOT NULL,
    overall_score   DOUBLE,
    task_success_score DOUBLE,
    relevance_score DOUBLE,
    hallucination_score DOUBLE,
    consistency_score DOUBLE,
    source_run_id   VARCHAR,
    commit_hash     VARCHAR,
    created_at      TIMESTAMP DEFAULT current_timestamp,
    updated_at      TIMESTAMP DEFAULT current_timestamp
);
"""


class Database:
    """DuckDB database manager for LLM Quality Gate."""

    def __init__(self, db_path: str = "llmqg.duckdb"):
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self.conn.execute(SCHEMA_SQL)
        self.conn.execute("ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS regression_detected BOOLEAN DEFAULT FALSE")
        self.conn.execute("ALTER TABLE evaluation_runs ADD COLUMN IF NOT EXISTS regression_summary JSON")
        logger.info(f"Database schema initialized at {self.db_path}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params=None):
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def fetchall(self, query: str, params=None):
        result = self.execute(query, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def fetchone(self, query: str, params=None):
        result = self.execute(query, params)
        columns = [desc[0] for desc in result.description]
        row = result.fetchone()
        if row:
            return dict(zip(columns, row))
        return None
