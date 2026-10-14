# Patent Search — Installation and Setup Guide

## 1. Overview

This repository contains a Streamlit-based patent search and analysis prototype. It uses Python services for patent retrieval, deduplication, classification lookup, BM25 and embedding-based ranking, a MySQL database, and optional AI-assisted analysis.

This guide covers:

- Windows with WSL2 Ubuntu (the current development setup)
- A fresh native macOS installation
- Python environment setup
- MySQL setup and database restoration
- API credentials and optional local Mistral/Ollama support
- Classification data import
- Tests and startup

> **Prototype notice:** This guide describes installation, not validation of patentability or legal advice. Search-provider access, API credentials, classification imports, and ranking quality must be verified on each machine.

## 2. Before you begin

You will need:

- Git access to this repository
- Python 3.12 as the initial target, matching the development environment
- MySQL 8.x, if using the current MySQL storage layer
- Network access for package installation, model downloads, and any live patent-search providers
- API credentials for the selected search provider and optional AI features
- The project’s required classification archives and, if migrating existing data, the database dump

Python 3.12 is a starting point, not a claim that every dependency has been tested on every operating system. Validate the dependency installation and test suite on the target machine before treating that platform as supported.

## 3. Files and secrets

Keep these items separate:

- `.env`: local credentials and machine-specific settings. **Do not commit this file.**
- `requirements.txt`: Python dependencies.
- `patent_db.sql`: optional database backup for migration. Treat it as data, keep it out of Git, and transfer it securely.
- `data/classification/`: classification source archives used by import scripts.
- `versions/`: contains historical archives and sample files. Do not assume every file here is required at runtime.

The project has large classification archives. Before copying the entire `versions/` directory to another machine, determine which archives are required for the desired features and which are historical duplicates.

## 4. Windows with WSL2 Ubuntu

Run these commands **inside the Ubuntu/WSL terminal**, not in Windows PowerShell.

### 4.1 Install system packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential pkg-config
python3 --version
```

Confirm that Python is 3.12.x for the initial migration.

### 4.2 Clone the repository

Use the actual repository URL for your project:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <YOUR_REPOSITORY_URL> patent-search
cd patent-search
```

If the repository is already cloned, enter its directory and check the branch and working tree:

```bash
git status
git branch --show-current
```

Do not copy the development machine’s virtual environment to another machine. Create a fresh environment instead.

### 4.3 Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Before installation, review `requirements.txt` and add any confirmed missing direct dependencies. The current source import audit identified `lxml` and `ollama`, which are not in the supplied requirements list:

```text
lxml>=5.0
ollama>=0.3
```

These are suggested minimums and should be validated with the complete application. Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

If installation fails for `faiss-cpu`, PyTorch, or Sentence Transformers, retain the full error output and resolve the platform/version compatibility issue rather than installing arbitrary versions.

### 4.4 Install and configure MySQL

Install MySQL 8.x using the supported package instructions for your Ubuntu version, or use an existing MySQL server. Confirm that the server is running and that you can log in.

Create a dedicated application database user rather than using `root` for normal app access. The exact SQL commands depend on the local MySQL authentication configuration. For example, after logging in as an administrator:

```sql
CREATE USER 'patent_app'@'localhost' IDENTIFIED BY 'REPLACE_WITH_A_STRONG_PASSWORD';
CREATE DATABASE patent_db CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
GRANT ALL PRIVILEGES ON patent_db.* TO 'patent_app'@'localhost';
FLUSH PRIVILEGES;
```

If the database already exists, inspect it before running creation or restore commands. A dump may contain `CREATE DATABASE`, `USE`, and `DROP TABLE` statements.

### 4.5 Configure `.env`

Create a local `.env` in the repository root. Use the variable names supported by `app/config.py`:

```dotenv
# Search backend: auto, epo, uspto, or patentsview
PATENT_SEARCH_BACKEND=epo

# EPO OPS credentials, if using the EPO backend
EPO_CONSUMER_KEY=
EPO_CONSUMER_SECRET=

# Optional AI analysis
ANTHROPIC_API_KEY=

# Database connection
DB_HOST=localhost
DB_USER=patent_app
DB_PASSWORD=REPLACE_WITH_YOUR_PASSWORD
DB_NAME=patent_db

# Optional model and ranking settings
ANTHROPIC_MODEL=claude-sonnet-4-6
MISTRAL_MODEL=mistral
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
BM25_WEIGHT=0.3
COSINE_WEIGHT=0.7
TOP_K_RESULTS=10
```

Use credentials for the backend you actually intend to use. If selecting `epo`, supply valid EPO OPS credentials. If selecting `patentsview` or `uspto`, confirm the corresponding service implementation and current provider access requirements before relying on it. `auto` selects based on available configuration and may fail if no supported backend is configured.

The current settings also accept `LENS_API_TOKEN`, `ANTHROPIC_API_KEY`, `EPO_API_URL`, and other fields declared in `app/config.py`. Use the exact field names defined there. Do not paste real secrets into source control, issue trackers, or chat logs.

### 4.6 Restore existing data (optional)

If you are migrating the existing database, transfer `patent_db.sql` securely to the new machine. First inspect the dump header and its database/table statements. **A dump that contains `DROP TABLE` can overwrite existing data.**

If the dump is known to be the intended backup and its contents have been reviewed, restore it using the appropriate MySQL account. For a dump that targets `patent_db` internally:

```bash
mysql -u patent_app -p < /path/to/patent_db.sql
```

If the database user does not have the privileges needed by the dump, use an authorized database administrator account for the restore, then grant the application user access. Do not blindly restore over an existing database.

If starting from an empty database instead, the application’s `create_tables()` function creates the tables declared in `storage/database.py`. This does not by itself populate the classification catalogue.

### 4.7 Import classification data (when required)

The repository contains CPC classification archives and import scripts. Confirm the intended source version before importing.

The CPC scheme importer currently references:

```text
data/classification/CPCSchemeXML202608.zip
```

Run it from the repository root after confirming the database configuration:

```bash
python scripts/import_cpc_scheme.py
```

The definitions importer accepts an explicit archive path:

```bash
python scripts/import_cpc_definitions.py --help
python scripts/import_cpc_definitions.py --zip data/classification/FullCPCDefinitionXML202608.zip
```

Large imports can take time and require substantial disk space. Check the script help and database state before running an import. Do not import duplicate or historical versions unless you intend to retain them.

## 5. Fresh native macOS installation

The macOS setup should be performed natively. Do not copy `.venv` from WSL or another computer.

### 5.1 Install prerequisites

Install Git, Python 3.12, and MySQL 8.x using your preferred trusted package manager or the official installation instructions. If using Homebrew, first verify that Homebrew is installed, then install the required packages with the package names available for your macOS version.

Confirm:

```bash
python3 --version
git --version
mysql --version
```

Ensure the MySQL service is running and create the application database/user as described in the MySQL section.

### 5.2 Clone and install

```bash
mkdir -p ~/projects
cd ~/projects
git clone <YOUR_REPOSITORY_URL> patent-search
cd patent-search

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the same `.env` setup and database migration/restore procedure described above, adjusted for the local MySQL host and credentials.

### 5.3 Validate macOS compatibility

Do not assume that a successful Linux install guarantees a successful macOS install. In particular, validate:

- PyTorch and Sentence Transformers installation
- `faiss-cpu` installation for the Mac's CPU architecture
- First-time embedding model download and local cache permissions
- MySQL connectivity and database collation compatibility
- Classification imports and the complete test suite

If any dependency does not support the selected Python/macOS/CPU combination, record the exact error and revise the dependency specification deliberately. Do not copy Linux wheels or the WSL virtual environment to macOS.

## 6. Run tests

Activate the project virtual environment first:

```bash
source .venv/bin/activate
python -m pytest -q tests/
```

The unit-test suite is the first validation step. Some scripts, such as `scripts/test_patentsview.py`, make live network requests and may fail because of provider availability, credentials, DNS, or rate limits. Treat live-provider checks separately from offline unit tests.

A successful test run does not prove that API credentials, database contents, classification data, or every live search path are configured correctly.

## 7. Launch Streamlit

From the repository root with the virtual environment activated:

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

If the app cannot start, check the traceback, `.env` values, database connectivity, and whether all required packages are installed in the active virtual environment.

## 8. First-run checklist

- [ ] Repository cloned and branch/commit recorded
- [ ] New virtual environment created on the target machine
- [ ] Requirements installed without unresolved errors
- [ ] `.env` created locally and excluded from Git
- [ ] Selected search backend and credentials configured
- [ ] MySQL connection verified
- [ ] Existing database restored, or empty schema created
- [ ] Required CPC scheme/definition data imported
- [ ] `python -m pytest -q tests/` passes
- [ ] Streamlit starts successfully
- [ ] A representative search and classification lookup have been manually checked
- [ ] AI-assisted features tested separately if enabled

## 9. Troubleshooting notes

### MySQL connection errors
Verify the MySQL service, host, port, username, password, database name, and user grants. Check `DB_URL` if you have overridden the generated connection URL.

### Search returns no results
Confirm the selected backend, credentials, endpoint access, and provider response. A running UI does not guarantee that the external search provider is available.

### Embedding model downloads fail
The first run may need network access to download the configured Hugging Face model. Confirm connectivity, disk space, and cache permissions.

### Classification lookup is empty
Confirm that the required archives exist at the configured paths and that the corresponding import completed successfully in the database configured by `.env`.

### Dependency installation differs by platform
Record the OS version, CPU architecture, Python version, package name, and complete installation error. Resolve it with a tested dependency/version change, then update the requirements specification.

## 10. Known limitations and follow-up before release

This guide is a migration baseline, not yet a certified one-command installer. Before treating the project as portable, the repository should also have:

1. A reviewed `.env.example` with safe placeholders.
2. A complete direct-dependency audit across all runtime modules.
3. A tested dependency lock or constraints file.
4. A documented and tested database backup/restore process.
5. A clear inventory of required classification archives and optional historical files.
6. Separate offline unit tests and live-provider integration checks.
7. A clean-install test on both WSL2 Ubuntu and the intended macOS version/CPU architecture.
