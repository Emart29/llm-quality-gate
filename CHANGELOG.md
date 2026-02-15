# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-15

### Added
- Core config discovery utilities in `core/` with upward project search and explicit config path support.
- CLI standalone evaluation fallback when dashboard API is not running.
- Standardized CLI exit code behavior: `0` pass, `1` gate/evaluation failure, `2` config errors.

### Changed
- Standardized default configuration filename to `llmq.yaml` across CLI, API, service, and factory layers.
- API now returns user-facing `400` errors for configuration problems instead of surfacing generic `500` responses.
- Packaging now includes `core*` modules.

### Fixed
- Dashboard/API config resolution no longer depends on being launched from repository root.
- Missing config now reports: `No llmq.yaml found. Run \`llmq init\`.`
