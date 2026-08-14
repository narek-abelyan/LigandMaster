# LigandMaster

An interactive Dash dashboard for exploring molecular docking / screening results. Upload a CSV of molecules and get an instant interactive workspace: sort and filter thousands of rows, plot any numeric property against another, view distributions, render 2D structures on click, compute extra molecular descriptors on demand, and build a shortlist of interesting molecules to export.

Your CSV needs at minimum an **`ID`** and a **`SMILES`** column — everything else (docking scores, molecular weight, custom properties, whatever your pipeline produces) is picked up automatically as long as it's numeric.

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

## Running locally (conda)

```bash
git clone https://github.com/narek-abelyan/LigandMaster.git
cd LigandMaster

conda create -n ligandmaster python=3.11
conda activate ligandmaster
pip install -r requirements.txt

python app.py
```

Open **http://localhost:8054** — you'll land on a login page. Enter the username and password from `APP_USERS` (see below) to get in.

### Setting the username and password

Login credentials come from the `APP_USERS` environment variable, in the format `user:password`, with commas to separate multiple accounts. Set it *before* running `python app.py`, in the same terminal:

```bash
export APP_USERS="narek:mypassword"
# or several accounts:
export APP_USERS="narek:pass1,alice:pass2"

python app.py
```

If `APP_USERS` isn't set, the app quietly falls back to `admin:changeme` — fine for a quick local check, but change it for anything you'd share with others. To avoid retyping it every session, add the `export APP_USERS=...` line to your `~/.bashrc` (or `~/.zshrc`) instead.

## Tech stack

Dash · Plotly · dash-bootstrap-components · RDKit · Pandas · Flask (auth & sessions)

---
Designed by N. Abelyan
