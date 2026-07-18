# Security Policy

## Supported Versions

LMC-5 is currently alpha. Security fixes target the main branch until stable
release branches exist.

## Reporting a Vulnerability

Please open a private advisory or contact the maintainers through the repository
security channel if available. Do not paste live credentials, production logs,
or private user data into public issues.

## Scope

Security-sensitive areas include:

- Redaction failures.
- Unsafe prompt-injection surfaces.
- Accidental credential persistence.
- Unsafe automatic memory mutation.
- Import/export handling that leaks private data.

## Defaults

The original local core package:

- Makes no network calls.
- Stores data locally in SQLite.
- Redacts recall output by default in the CLI.
- Keeps patrol checks read-only.

These defaults do not make LMC-5 a complete secrets-management or DLP system.
Treat it as an agent-memory safety layer, not as a vault.

The `lmc5_web` deployment is intentionally networked. It protects MCP with
OAuth 2.1, protects the import API with `LMC5_ACCESS_TOKEN`, skips known
credential files, and applies privacy gates before recall. Operate it behind
HTTPS, keep one application replica, restrict database access, and back up both
PostgreSQL and the `/data` OAuth store.
