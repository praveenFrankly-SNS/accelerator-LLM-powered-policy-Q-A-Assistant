# Contributing to LLM-Powered Policy Q&A Assistant

Thank you for your interest in contributing to this Databricks accelerator.

---

## Reporting Issues

Open a GitHub Issue with:
- **Title**: Short description of the problem
- **Environment**: Databricks Runtime version, cloud provider (AWS/Azure/GCP), region
- **Steps to reproduce**: Exact notebook and cell, widget values used
- **Expected vs actual behavior**
- **Error message / stack trace** (redact any credentials or PII)

---

## Contributing Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Follow the coding standards below
4. Add or update tests in `/tests/`
5. Run `pytest tests/ --cov=. --cov-report=term-missing` and ensure ≥70% coverage
6. Open a Pull Request against `main` with a clear description

### Coding Standards

- All notebooks must have the standard header markdown cell (accelerator name, version, author, runtime, prerequisites)
- Zero hardcoded catalog names, schema names, credentials, or cluster IDs — use `dbutils.widgets` or `config.py`
- All secrets via `dbutils.secrets.get()` — never in code or comments
- Structured logging (no bare `print()` in production paths — use the audit log pattern)
- Every pipeline stage must write to `pipeline_audit_log` on both success and failure
- Failed records go to a quarantine table — never silently dropped
- New notebooks must be added to `RUNME.py` and `databricks.yml`

### Testing

```bash
# Run unit tests
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_chunking.py -v
```

---

## Contact

**SNS Square Accelerators Team**
- Email: accelerators@snssquare.com
- GitHub Issues: preferred for bug reports and feature requests
