# LigandMaster

An interactive Dash dashboard for exploring molecular docking / screening results — upload a CSV of molecules, filter and sort them, visualize property distributions and correlations, and inspect 2D structures on the fly.

**`app.py` is the current, actively maintained dashboard and the one this README covers.** Other top-level scripts in this repo (`ClastMaster.py`, `LigandMaster1.2.py`, `LigandMaster1.3.py`) are earlier iterations kept for reference and aren't part of this app.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Dash](https://img.shields.io/badge/dash-3.2-informational)
![RDKit](https://img.shields.io/badge/rdkit-2025.9-success)

## Features

- **CSV upload** — bring your own dataset (`ID`, `SMILES` required; any number of extra numeric columns).
- **Interactive table** — sort, filter, and choose which columns to display.
- **Scatter plot & histograms** — pick any numeric columns for the X/Y axes and distributions, with optional KDE overlay.
- **2D structure viewer** — click a row to render its molecule via RDKit.
- **On-demand property calculation** — compute descriptors (MolWt, LogP, QED, TPSA, and more) for molecules that don't already have them.
- **Selected Molecules workspace** — build a shortlist, dedupe by canonical SMILES, round values, and export to CSV.
- **Login-protected & multi-user safe** — each logged-in session gets its own isolated copy of the data, so one user's upload never affects another's.

## Running locally

```bash
git clone https://github.com/narek-abelyan/LigandMaster.git
cd LigandMaster
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export APP_USERS="youruser:yourpassword"
export SECRET_KEY="some-random-string"
python app.py
```

Open **http://localhost:8054** and log in.

`APP_USERS` accepts multiple accounts as `user1:pass1,user2:pass2`. Without it, the app falls back to `admin:changeme` — fine for local testing, not for anything public.

## Deployment

Deployed on [Render](https://render.com) via `gunicorn app:server`. Set `APP_USERS` and `SECRET_KEY` as environment variables in the service settings — the app won't have sane defaults for either in production otherwise.

## Tech stack

Dash · Plotly · dash-bootstrap-components · RDKit · Pandas · Flask (auth & sessions)

---
Designed by N. Abelyan
