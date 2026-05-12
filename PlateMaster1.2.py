from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable
import importlib.util
import os
import re
import shutil
import math
import subprocess
import tempfile
import zipfile

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover
    get_script_run_ctx = None

EXCEL_EXTENSIONS = (".xlsx", ".xls", ".xlsm")
OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")





def _available_excel_engines_for_suffix(suffix: str) -> list[str]:
    suffix = suffix.lower()
    engines = ["openpyxl", "xlrd"] if suffix in {".xlsx", ".xlsm", "xlsx", "xlsm"} else ["xlrd", "openpyxl"]
    if importlib.util.find_spec("python_calamine") is not None:
        # Calamine can read both .xls and .xlsx and is often more tolerant.
        engines.insert(0, "calamine")
    return engines


def _try_libreoffice_convert_bytes_to_xlsx(content: bytes, filename: str = "input.xls") -> bytes | None:
    office_bin = shutil.which("libreoffice") or shutil.which("soffice")
    if office_bin is None:
        return None

    suffix = Path(filename).suffix or ".xls"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(content)

        cmd = [office_bin, "--headless", "--convert-to", "xlsx", "--outdir", tmpdir, str(input_path)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            return None

        output_path = input_path.with_suffix(".xlsx")
        if not output_path.exists():
            matches = list(Path(tmpdir).glob("*.xlsx"))
            if not matches:
                return None
            output_path = matches[0]

        return output_path.read_bytes()

def discover_excel_files(folder: str) -> list[Path]:
    base = Path(folder).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        return []
    patterns = ("*.xlsx", "*.xls", "*.xlsm")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return sorted(files)




def _read_spreadsheet_flexible_from_bytes(content: bytes, filename: str = "") -> pd.DataFrame:
    errors: list[str] = []
    suffix = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ""

    engine_attempts = _available_excel_engines_for_suffix(suffix)
    for engine in engine_attempts:
        try:
            return pd.read_excel(BytesIO(content), engine=engine)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")

    # Fallback for instrument exports mislabeled as .xls but containing delimited text.
    if not content.startswith(OLE_SIGNATURE):
        text = content.decode("utf-8", errors="ignore")
        for sep in ["   ", ";", ","]:
            try:
                frame = pd.read_csv(StringIO(text), sep=sep)
                if frame.shape[1] >= 2:
                    return frame
            except Exception as exc:
                errors.append(f"csv sep={sep!r}: {exc}")

    converted_xlsx = _try_libreoffice_convert_bytes_to_xlsx(content, filename or "input.xls")
    if converted_xlsx is not None:
        try:
            return pd.read_excel(BytesIO(converted_xlsx), engine="openpyxl")
        except Exception as exc:
            errors.append(f"libreoffice-convert: {exc}")

    raise ValueError(
        "Unable to parse spreadsheet content. "
        "If this is old .xls, ensure python-calamine is installed and optionally LibreOffice for auto-convert. Attempts: "
        + " | ".join(errors)
    )


def _read_spreadsheet_flexible_from_path(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    suffix = path.suffix.lower()
    engine_attempts = _available_excel_engines_for_suffix(suffix)

    for engine in engine_attempts:
        try:
            return pd.read_excel(path, engine=engine)
        except Exception as exc:
            errors.append(f"{engine}: {exc}")

    for sep in ["   ", ";", ","]:
        try:
            frame = pd.read_csv(path, sep=sep)
            if frame.shape[1] >= 2:
                return frame
        except Exception as exc:
            errors.append(f"csv sep={sep!r}: {exc}")

    converted_xlsx = _try_libreoffice_convert_bytes_to_xlsx(path.read_bytes(), path.name)
    if converted_xlsx is not None:
        try:
            return pd.read_excel(BytesIO(converted_xlsx), engine="openpyxl")
        except Exception as exc:
            errors.append(f"libreoffice-convert: {exc}")

    raise ValueError(
        f"Unable to parse file {path.name}. "
        "Install python-calamine and optionally LibreOffice for conversion. Attempts: "
        + " | ".join(errors)
    )

def sanitize_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Empty table")

    first_col = frame.columns[0]
    frame = frame.rename(columns={first_col: "row_id"})
    frame["row_id"] = frame["row_id"].astype(str).str.strip()
    frame = frame[frame["row_id"].str.len() > 0].copy()

    numeric_cols: list[str] = []
    for column in frame.columns[1:]:
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        if numeric_series.notna().any():
            col_name = str(column).strip()
            frame[col_name] = numeric_series
            numeric_cols.append(col_name)

    if not numeric_cols:
        raise ValueError("No numeric columns detected")

    result = frame[["row_id", *numeric_cols]].dropna(how="all", subset=numeric_cols)
    if result.empty:
        raise ValueError("No numeric rows detected")
    return result


def load_table_from_path(path: Path) -> pd.DataFrame:
    return sanitize_table(_read_spreadsheet_flexible_from_path(path))


def load_table_from_bytes(content: bytes, filename: str = "") -> pd.DataFrame:
    return sanitize_table(_read_spreadsheet_flexible_from_bytes(content, filename))




def _column_sort_key(label: str) -> tuple[int, float | str]:
    label_str = str(label).strip()
    try:
        return (0, float(label_str.replace(",", ".")))
    except ValueError:
        return (1, label_str)


def _sort_columns_natural(labels: list[str]) -> list[str]:
    return sorted(labels, key=_column_sort_key)


def _extract_time_from_filename(filename: str) -> float | None:
    stem = Path(filename).stem
    matches = re.findall(r"(\d+(?:[\.,]\d+)?)", stem)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "."))
    except ValueError:
        return None




def _format_time_label(value: float | int | str | None) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
        text = f"{num:.6f}".rstrip("0").rstrip(".")
        return text
    except Exception:
        return str(value)

def _remove_outliers(series: pd.Series, method: str, strictness: float) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty or method == "None":
        return s

    if method == "IQR":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return s
        lower = q1 - strictness * iqr
        upper = q3 + strictness * iqr
        return s[(s >= lower) & (s <= upper)]

    if method == "Z-score":
        std = s.std(ddof=0)
        if std == 0:
            return s
        z = (s - s.mean()).abs() / std
        return s[z <= strictness]

    if method == "MAD":
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0:
            return s
        modified_z = 0.6745 * (s - median).abs() / mad
        return s[modified_z <= strictness]

    return s

def aggregate_rows(
    table: pd.DataFrame,
    selected_rows: list[str],
    selected_columns: list[str],
    func: Callable[[pd.Series], float],
    outlier_method: str = "None",
    outlier_strictness: float = 1.5,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, dict[str, float]], dict[str, list[float]]]:
    filtered = table[table["row_id"].isin(selected_rows)]
    if filtered.empty:
        raise ValueError("Selected rows not found in table")

    removed_rows: list[dict[str, float | str]] = []
    values: dict[str, float] = {}
    column_stats: dict[str, dict[str, float]] = {}
    source_values: dict[str, list[float]] = {}
    for column in selected_columns:
        if column not in filtered.columns:
            continue
        original_series = filtered[column].dropna()
        cleaned_series = _remove_outliers(original_series, outlier_method, outlier_strictness)

        removed_indices = original_series.index.difference(cleaned_series.index)
        for idx in removed_indices:
            removed_rows.append({
                "row_id": str(filtered.loc[idx, "row_id"]),
                "column": str(column),
                "value": float(original_series.loc[idx]),
            })

        if cleaned_series.empty:
            continue
        values[column] = func(cleaned_series)
        source_values[column] = [float(x) for x in cleaned_series.tolist()]

        n = float(len(cleaned_series))
        std = float(cleaned_series.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        ci95 = float(1.96 * sem)
        q1 = float(cleaned_series.quantile(0.25))
        q3 = float(cleaned_series.quantile(0.75))
        iqr_half = float((q3 - q1) / 2.0)
        mad = float((cleaned_series - cleaned_series.median()).abs().median())
        column_stats[column] = {
            "std": std,
            "sem": sem,
            "ci95": ci95,
            "iqr_half": iqr_half,
            "mad": mad,
        }

    if not values:
        raise ValueError("No values for selected columns")

    return pd.DataFrame([values], index=["aggregated"]), removed_rows, column_stats, source_values






def no_aggregate_rows(
    table: pd.DataFrame,
    selected_rows: list[str],
    selected_columns: list[str],
    outlier_method: str = "None",
    outlier_strictness: float = 1.5,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, dict[str, float]], dict[str, list[float]]]:
    filtered = table[table["row_id"].isin(selected_rows)]
    if filtered.empty:
        raise ValueError("Selected rows not found in table")

    removed_rows: list[dict[str, float | str]] = []
    values: dict[str, float] = {}
    column_stats: dict[str, dict[str, float]] = {}
    source_values: dict[str, list[float]] = {}

    for column in selected_columns:
        if column not in filtered.columns:
            continue
        original_series = pd.to_numeric(filtered[column], errors="coerce").dropna()
        cleaned_series = _remove_outliers(original_series, outlier_method, outlier_strictness)

        removed_indices = original_series.index.difference(cleaned_series.index)
        for idx in removed_indices:
            removed_rows.append({
                "row_id": str(filtered.loc[idx, "row_id"]),
                "column": str(column),
                "value": float(original_series.loc[idx]),
            })

        if cleaned_series.empty:
            continue

        for i, val in enumerate(cleaned_series.tolist(), start=1):
            values[f"{column}__{i}"] = float(val)

        source_values[column] = [float(x) for x in cleaned_series.tolist()]
        n = float(len(cleaned_series))
        std = float(cleaned_series.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        ci95 = float(1.96 * sem)
        q1 = float(cleaned_series.quantile(0.25))
        q3 = float(cleaned_series.quantile(0.75))
        iqr_half = float((q3 - q1) / 2.0)
        mad = float((cleaned_series - cleaned_series.median()).abs().median())
        column_stats[column] = {"std": std, "sem": sem, "ci95": ci95, "iqr_half": iqr_half, "mad": mad}

    if not values:
        raise ValueError("No values for selected columns")

    return pd.DataFrame([values], index=["aggregated"]), removed_rows, column_stats, source_values


def aggregate_by_groups(
    table: pd.DataFrame,
    group_map: dict[str, list[tuple[str, str]]],
    func: Callable[[pd.Series], float],
    outlier_method: str = "None",
    outlier_strictness: float = 1.5,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, dict[str, float]], dict[str, list[float]]]:
    removed_rows: list[dict[str, float | str]] = []
    values: dict[str, float] = {}
    group_stats: dict[str, dict[str, float]] = {}
    source_values: dict[str, list[float]] = {}

    table_indexed = table.set_index("row_id", drop=False)

    for group_name, members in group_map.items():
        collected: list[tuple[str, str, float]] = []
        for row_id, col in members:
            if row_id in table_indexed.index and col in table_indexed.columns:
                v = pd.to_numeric(table_indexed.at[row_id, col], errors="coerce")
                if pd.notna(v):
                    collected.append((row_id, col, float(v)))

        if not collected:
            continue

        original_series = pd.Series([x[2] for x in collected])
        cleaned_series = _remove_outliers(original_series, outlier_method, outlier_strictness)

        removed_indices = original_series.index.difference(cleaned_series.index)
        for idx in removed_indices:
            row_id, col, val = collected[int(idx)]
            removed_rows.append({"row_id": row_id, "column": str(col), "group": group_name, "value": float(val)})

        if cleaned_series.empty:
            continue

        values[group_name] = func(cleaned_series)
        source_values[group_name] = [float(x) for x in cleaned_series.tolist()]
        n = float(len(cleaned_series))
        std = float(cleaned_series.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        ci95 = float(1.96 * sem)
        q1 = float(cleaned_series.quantile(0.25))
        q3 = float(cleaned_series.quantile(0.75))
        iqr_half = float((q3 - q1) / 2.0)
        mad = float((cleaned_series - cleaned_series.median()).abs().median())
        group_stats[group_name] = {"std": std, "sem": sem, "ci95": ci95, "iqr_half": iqr_half, "mad": mad}

    if not values:
        raise ValueError("No values for selected groups")

    return pd.DataFrame([values], index=["aggregated"]), removed_rows, group_stats, source_values


def aggregate_columns_by_row(
    table: pd.DataFrame,
    selected_rows: list[str],
    selected_columns: list[str],
    func: Callable[[pd.Series], float],
    outlier_method: str = "None",
    outlier_strictness: float = 1.5,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, dict[str, float]], dict[str, list[float]]]:
    filtered = table[table["row_id"].isin(selected_rows)]
    if filtered.empty:
        raise ValueError("Selected rows not found in table")

    removed_rows: list[dict[str, float | str]] = []
    values: dict[str, float] = {}
    row_stats: dict[str, dict[str, float]] = {}
    source_values: dict[str, list[float]] = {}

    for _, row in filtered.iterrows():
        row_id = str(row["row_id"])
        row_series = pd.to_numeric(row[selected_columns], errors="coerce").dropna()
        if row_series.empty:
            continue

        cleaned_series = _remove_outliers(row_series, outlier_method, outlier_strictness)
        removed_indices = row_series.index.difference(cleaned_series.index)
        for col in removed_indices:
            removed_rows.append({"row_id": row_id, "column": str(col), "value": float(row_series.loc[col])})

        if cleaned_series.empty:
            continue

        values[row_id] = func(cleaned_series)
        source_values[row_id] = [float(x) for x in cleaned_series.tolist()]
        n = float(len(cleaned_series))
        std = float(cleaned_series.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / math.sqrt(n)) if n > 1 else 0.0
        ci95 = float(1.96 * sem)
        q1 = float(cleaned_series.quantile(0.25))
        q3 = float(cleaned_series.quantile(0.75))
        iqr_half = float((q3 - q1) / 2.0)
        mad = float((cleaned_series - cleaned_series.median()).abs().median())
        row_stats[row_id] = {"std": std, "sem": sem, "ci95": ci95, "iqr_half": iqr_half, "mad": mad}

    if not values:
        raise ValueError("No values for selected rows")

    return pd.DataFrame([values], index=["aggregated"]), removed_rows, row_stats, source_values


def _basic_stat_metrics(values: list[float]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame([{"Metric": "No data", "Value": "N/A"}])

    s = pd.Series(values, dtype=float)
    n = len(s)
    mean = float(s.mean())
    median = float(s.median())
    mode_vals = s.mode()
    mode = float(mode_vals.iloc[0]) if not mode_vals.empty else float("nan")
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
    mad = float((s - s.median()).abs().median())
    min_v = float(s.min())
    max_v = float(s.max())
    ci95 = float(1.96 * sem)
    iqr_half = float((s.quantile(0.75) - s.quantile(0.25)) / 2.0)

    # Approximate two-sided p-value for mean != 0 using normal approximation.
    if n > 1 and sem > 0:
        z = abs(mean / sem)
        p_value = float(math.erfc(z / math.sqrt(2.0)))
    else:
        p_value = float("nan")

    rows = [
        {"Metric": "n", "Value": n},
        {"Metric": "mean", "Value": mean},
        {"Metric": "median", "Value": median},
        {"Metric": "mode", "Value": mode},
        {"Metric": "sd", "Value": sd},
        {"Metric": "sem", "Value": sem},
        {"Metric": "mad", "Value": mad},
        {"Metric": "min", "Value": min_v},
        {"Metric": "max", "Value": max_v},
        {"Metric": "95CI", "Value": ci95},
        {"Metric": "IQR/2", "Value": iqr_half},
        {"Metric": "p-value (mean != 0, approx)", "Value": p_value},
    ]
    return pd.DataFrame(rows)


def _welch_t_test(values_a: list[float], values_b: list[float]) -> pd.DataFrame:
    if len(values_a) < 2 or len(values_b) < 2:
        return pd.DataFrame(
            [{"Metric": "T-test", "Value": "Need at least 2 values in each group"}]
        )

    a = pd.Series(values_a, dtype=float)
    b = pd.Series(values_b, dtype=float)
    n1 = float(len(a))
    n2 = float(len(b))
    m1 = float(a.mean())
    m2 = float(b.mean())
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    se = math.sqrt((v1 / n1) + (v2 / n2))
    if se == 0:
        return pd.DataFrame([{"Metric": "T-test", "Value": "Undefined (zero variance)"}])

    t_stat = (m1 - m2) / se
    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2 / (n1 - 1.0)) + ((v2 / n2) ** 2 / (n2 - 1.0))
    dof = num / den if den > 0 else float("nan")
    p_approx = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))

    return pd.DataFrame(
        [
            {"Metric": "Group A n", "Value": int(n1)},
            {"Metric": "Group B n", "Value": int(n2)},
            {"Metric": "Group A mean", "Value": m1},
            {"Metric": "Group B mean", "Value": m2},
            {"Metric": "Welch t", "Value": float(t_stat)},
            {"Metric": "df (Welch)", "Value": float(dof)},
            {"Metric": "p-value (2-sided, approx)", "Value": p_approx},
        ]
    )


def _welch_p_value(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")

    a = pd.Series(values_a, dtype=float)
    b = pd.Series(values_b, dtype=float)
    n1 = float(len(a))
    n2 = float(len(b))
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    se = math.sqrt((v1 / n1) + (v2 / n2))
    if se == 0:
        return float("nan")

    t_stat = (float(a.mean()) - float(b.mean())) / se
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def _build_excel_report(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = re.sub(r"[\[\]\*\:/\\\?]", "_", sheet_name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe_sheet or "Sheet1")
    output.seek(0)
    return output.getvalue()


def _to_numeric_series(values: pd.Series | list[float] | np.ndarray) -> pd.Series:
    return pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").dropna()


AGGREGATIONS: dict[str, Callable[[pd.Series | list[float] | np.ndarray], float] | None] = {
    "None": None,
    "Mean": lambda s: float(_to_numeric_series(s).mean()),
    "Median": lambda s: float(_to_numeric_series(s).median()),
    "Min": lambda s: float(_to_numeric_series(s).min()),
    "Max": lambda s: float(_to_numeric_series(s).max()),
    "Std": lambda s: float(_to_numeric_series(s).std(ddof=0)),
    "Sum": lambda s: float(_to_numeric_series(s).sum()),
}




def _gradient_style(df: pd.DataFrame, mode: str = "Global scale", palette: str = "Rose"):
    if df.empty:
        return df
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if str(c).strip().lower() != "time"]
    if not numeric_cols:
        return df

    try:
        values = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        style_map = pd.DataFrame("", index=df.index, columns=df.columns)

        if mode == "Per-column scale":
            mins = values.min(axis=0)
            spans = (values.max(axis=0) - mins).replace(0, pd.NA)
            ratios = values.sub(mins, axis=1).div(spans, axis=1)
        elif mode == "Per row scale":
            mins = values.min(axis=1)
            spans = (values.max(axis=1) - mins).replace(0, pd.NA)
            ratios = values.sub(mins, axis=0).div(spans, axis=0)
        else:
            global_min = values.min().min()
            global_max = values.max().max()
            span = (global_max - global_min) if pd.notna(global_max) and pd.notna(global_min) else pd.NA
            if pd.isna(span) or span == 0:
                ratios = values * 0 + 0.5
            else:
                ratios = (values - global_min) / span

        palette_rgb: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
            "Rose": ((255.0, 246.0, 250.0), (255.0, 92.0, 175.0)),
            "Pink-Violet": ((255.0, 244.0, 252.0), (220.0, 120.0, 230.0)),
            "Purple": ((248.0, 244.0, 255.0), (171.0, 71.0, 188.0)),
            "Blue": ((240.0, 248.0, 255.0), (66.0, 135.0, 245.0)),
            "Green": ((243.0, 252.0, 245.0), (52.0, 168.0, 83.0)),
            "Orange": ((255.0, 248.0, 240.0), (245.0, 124.0, 0.0)),
            "Teal": ((240.0, 252.0, 252.0), (0.0, 137.0, 123.0)),
            "Gray": ((250.0, 250.0, 250.0), (97.0, 97.0, 97.0)),
        }
        start_rgb, end_rgb = palette_rgb.get(palette, palette_rgb["Rose"])

        for row_idx in ratios.index:
            for col in ratios.columns:
                ratio = ratios.loc[row_idx, col]
                if pd.isna(ratio):
                    continue
                ratio = max(0.0, min(1.0, float(ratio)))
                r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
                style_map.loc[row_idx, col] = f"background-color: rgb({r}, {g}, {b}); color: #111;"

        return df.style.apply(lambda _: style_map, axis=None)
    except Exception:
        return df


def run_dashboard() -> None:
    st.set_page_config(page_title="PlateMaster - Dashboard for Plate Analysis", layout="wide")
    st.title("PlateMaster - Dashboard for Plate Analysis")
    st.caption(
        "Load Excel files from server path OR upload from your local PC, choose rows/columns, aggregate, and plot across files."
    )


    if "loaded_tables" not in st.session_state:
        st.session_state.loaded_tables = {}
    if "load_errors" not in st.session_state:
        st.session_state.load_errors = []
    if "analysis_ready" not in st.session_state:
        st.session_state.analysis_ready = False
    if "analysis_presets" not in st.session_state:
        st.session_state.analysis_presets = {}

    loaded_tables: dict[str, pd.DataFrame] = dict(st.session_state.loaded_tables)
    errors: list[str] = list(st.session_state.load_errors)

    with st.sidebar:
        st.header("1) Data source")
        source_mode = st.radio("Source", ["Server folder path", "Local PC upload"], index=1)

        if source_mode == "Server folder path":
            folder_path = st.text_input("Server folder path with Excel files", value="")
            load_folder_clicked = st.button("Load folder")
            if load_folder_clicked:
                loaded_tables = {}
                errors = []
                for file in discover_excel_files(folder_path):
                    try:
                        loaded_tables[file.name] = load_table_from_path(file)
                    except Exception as exc:
                        errors.append(f"{file.name}: {exc}")
                st.session_state.loaded_tables = loaded_tables
                st.session_state.load_errors = errors
                st.session_state.analysis_ready = False

        else:
            uploaded_files = st.file_uploader(
                "Upload Excel files from your local PC",
                type=["xlsx", "xls", "xlsm"],
                accept_multiple_files=True,
            )
            uploaded_zip = st.file_uploader(
                "Or upload ONE ZIP with many Excel files",
                type=["zip"],
                accept_multiple_files=False,
            )
            load_uploads_clicked = st.button("Load uploaded files")

            if load_uploads_clicked:
                loaded_tables = {}
                errors = []
                for uploaded in uploaded_files or []:
                    try:
                        loaded_tables[uploaded.name] = load_table_from_bytes(uploaded.getvalue(), uploaded.name)
                    except Exception as exc:
                        errors.append(f"{uploaded.name}: {exc}")

                if uploaded_zip is not None:
                    try:
                        with zipfile.ZipFile(BytesIO(uploaded_zip.getvalue())) as archive:
                            for member in archive.namelist():
                                lower = member.lower()
                                if not lower.endswith(EXCEL_EXTENSIONS) or member.endswith("/"):
                                    continue
                                try:
                                    loaded_tables[Path(member).name] = load_table_from_bytes(archive.read(member), member)
                                except Exception as exc:
                                    errors.append(f"{member}: {exc}")
                    except Exception as exc:
                        errors.append(f"ZIP error: {exc}")

                st.session_state.loaded_tables = loaded_tables
                st.session_state.load_errors = errors
                st.session_state.analysis_ready = False

    if errors:
        st.warning("Some files were skipped:\n- " + "\n- ".join(errors))

    if not loaded_tables:
        st.info("Choose data source and click a load button (**Load folder** or **Load uploaded files**).")
        return

    color_scale_mode = st.session_state.get("color_scale_mode", "Global scale")
    color_palette = st.session_state.get("color_palette", "Rose")

    st.success(f"Loaded {len(loaded_tables)} Excel file(s)")

    representative_loaded_name = sorted(loaded_tables.keys())[0]
    st.header("Representative input table")
    st.subheader(f"Source: {representative_loaded_name}")
    st.dataframe(
        _gradient_style(loaded_tables[representative_loaded_name], color_scale_mode, color_palette),
        width="stretch",
    )

    all_rows = sorted({row for table in loaded_tables.values() for row in table["row_id"].astype(str).tolist()})
    all_columns = _sort_columns_natural(sorted({str(c) for table in loaded_tables.values() for c in table.columns if c != "row_id"}))

    with st.sidebar:
        st.header("2) Selection")
        layout_mode = st.selectbox(
            "Aggregation layout",
            options=["By original columns", "By original rows", "By plate groups (template)"],
            index=0,
            key="agg_layout_mode",
        )

        selected_rows = st.multiselect("Rows", options=all_rows, default=all_rows, key="selected_rows")
        selected_columns = st.multiselect("Columns", options=all_columns, default=all_columns, key="selected_columns")
        selected_columns = _sort_columns_natural(selected_columns)
        agg_options = list(AGGREGATIONS.keys())
        agg_default_index = agg_options.index("Mean") if "Mean" in agg_options else 0
        agg_name = st.selectbox("Aggregation", options=agg_options, index=agg_default_index, key="agg_name")

        st.subheader("Outlier filtering")
        outlier_method = st.selectbox("Method", options=["None", "IQR", "Z-score", "MAD"], index=0, key="outlier_method")
        outlier_strictness = 1.5
        if outlier_method == "IQR":
            outlier_strictness = st.slider("IQR multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="iqr_multiplier")
        elif outlier_method == "Z-score":
            outlier_strictness = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="zscore_threshold")
        elif outlier_method == "MAD":
            outlier_strictness = st.slider("MAD threshold", min_value=1.0, max_value=8.0, value=3.5, step=0.1, key="mad_threshold")

        color_controls_left, color_controls_right = st.columns(2)
        with color_controls_left:
            st.selectbox(
                "Color scale mode",
                options=["Global scale", "Per-column scale", "Per row scale"],
                key="color_scale_mode",
            )
        with color_controls_right:
            st.selectbox(
                "Color palette",
                options=["Rose", "Pink-Violet", "Purple", "Blue", "Green", "Orange", "Teal", "Gray"],
                key="color_palette",
            )
        color_scale_mode = st.session_state.color_scale_mode
        color_palette = st.session_state.color_palette

        st.subheader("Presets")
        preset_name = st.text_input("Preset name", value="", key="preset_name")
        preset_names = sorted(st.session_state.analysis_presets.keys())
        preset_to_load = st.selectbox(
            "Load preset",
            options=[""] + preset_names,
            format_func=lambda x: "Select preset..." if x == "" else x,
            key="preset_to_load",
        )
        preset_col1, preset_col3 = st.columns([3, 2])
        with preset_col1:
            save_preset_clicked = st.button("Save preset now", use_container_width=True)
        with preset_col3:
            load_preset_clicked = st.button("Apply preset", use_container_width=True)

        if save_preset_clicked:
            if not preset_name.strip():
                st.warning("Enter preset name before saving.")
            else:
                st.session_state.analysis_presets[preset_name.strip()] = {
                    "agg_layout_mode": layout_mode,
                    "selected_rows": selected_rows,
                    "selected_columns": selected_columns,
                    "agg_name": agg_name,
                    "outlier_method": outlier_method,
                    "iqr_multiplier": st.session_state.get("iqr_multiplier", 1.5),
                    "zscore_threshold": st.session_state.get("zscore_threshold", 3.0),
                    "mad_threshold": st.session_state.get("mad_threshold", 3.5),
                    "color_scale_mode": color_scale_mode,
                    "color_palette": color_palette,
                    "plot_columns": st.session_state.get("plot_columns", list(selected_columns)),
                    "plot_type": st.session_state.get("plot_type", "Line + points"),
                    "error_method": st.session_state.get("error_method", "None"),
                }
                st.success(f"Preset '{preset_name.strip()}' saved.")

        if load_preset_clicked:
            if not preset_to_load:
                st.warning("Please choose a preset to load.")
            else:
                preset = st.session_state.analysis_presets.get(preset_to_load, {})
                st.session_state.agg_layout_mode = preset.get("agg_layout_mode", "By original columns")
                st.session_state.selected_rows = [r for r in preset.get("selected_rows", all_rows) if r in all_rows]
                st.session_state.selected_columns = [c for c in preset.get("selected_columns", all_columns) if c in all_columns]
                st.session_state.agg_name = preset.get("agg_name", "Mean")
                st.session_state.outlier_method = preset.get("outlier_method", "None")
                st.session_state.iqr_multiplier = float(preset.get("iqr_multiplier", 1.5))
                st.session_state.zscore_threshold = float(preset.get("zscore_threshold", 3.0))
                st.session_state.mad_threshold = float(preset.get("mad_threshold", 3.5))
                st.session_state.color_scale_mode = preset.get("color_scale_mode", "Global scale")
                st.session_state.color_palette = preset.get("color_palette", "Rose")
                valid_plot_keys = set(all_columns) | set(all_rows)
                st.session_state.plot_columns = [c for c in preset.get("plot_columns", []) if c in valid_plot_keys]
                st.session_state.plot_type = preset.get("plot_type", "Line + points")
                st.session_state.error_method = preset.get("error_method", "None")
                st.success(f"Preset '{preset_to_load}' applied.")
                st.rerun()

        run_clicked = st.button("Run")
        if run_clicked:
            st.session_state.analysis_ready = True

    if not selected_rows or not selected_columns:
        st.warning("Please select at least one row and one column")
        return

    group_map: dict[str, list[tuple[str, str]]] = {}
    metric_labels = selected_columns
    if layout_mode == "By original rows":
        metric_labels = selected_rows
    elif layout_mode == "By plate groups (template)":
        st.header("Plate template grouping")
        st.caption("Enter group names into cells. Cells with the same label will be aggregated together.")
        template_df = pd.DataFrame("", index=selected_rows, columns=selected_columns)
        group_editor = st.data_editor(template_df, width="stretch", key="plate_group_template")
        for row_id in group_editor.index:
            for col in group_editor.columns:
                label = str(group_editor.loc[row_id, col]).strip()
                if label:
                    group_map.setdefault(label, []).append((str(row_id), str(col)))

        if not group_map:
            st.warning("Please define at least one group in plate template")
            return

        metric_labels = sorted(group_map.keys())

    if not st.session_state.analysis_ready:
        st.info("Press **Run** to start analysis with current settings.")
        return

    agg_func = AGGREGATIONS[agg_name]
    is_no_aggregation = agg_name == "None"
    aggregated_tables: dict[str, pd.DataFrame] = {}
    removed_outliers_log: list[dict[str, float | str | None]] = []
    total_points_per_column: dict[str, int] = {c: 0 for c in metric_labels}
    total_points_per_time: dict[float, int] = {}
    error_lookup: dict[tuple[str, str], dict[str, float]] = {}
    source_value_lookup: dict[tuple[str, str], list[float]] = {}

    for name, table in loaded_tables.items():
        try:
            filtered_for_stats = table[table["row_id"].isin(selected_rows)]
            file_time = _extract_time_from_filename(name)
            file_total_points = 0

            if layout_mode == "By plate groups (template)":
                table_indexed = table.set_index("row_id", drop=False)
                for group_name, members in group_map.items():
                    count = 0
                    for row_id, col in members:
                        if row_id in table_indexed.index and col in table_indexed.columns:
                            v = pd.to_numeric(table_indexed.at[row_id, col], errors="coerce")
                            if pd.notna(v):
                                count += 1
                    total_points_per_column[group_name] = total_points_per_column.get(group_name, 0) + count
                    file_total_points += count

                avg_table, removed_rows, column_stats, source_values = aggregate_by_groups(
                    table, group_map, agg_func, outlier_method, outlier_strictness
                )
            elif layout_mode == "By original rows":
                row_filtered = table[table["row_id"].isin(selected_rows)]
                for _, row in row_filtered.iterrows():
                    row_id = str(row["row_id"])
                    count = int(pd.to_numeric(row[selected_columns], errors="coerce").notna().sum())
                    total_points_per_column[row_id] = total_points_per_column.get(row_id, 0) + count
                    file_total_points += count

                avg_table, removed_rows, column_stats, source_values = aggregate_columns_by_row(
                    table, selected_rows, selected_columns, agg_func, outlier_method, outlier_strictness
                )
            else:
                for col in selected_columns:
                    if col in filtered_for_stats.columns:
                        count = int(filtered_for_stats[col].notna().sum())
                        total_points_per_column[col] = total_points_per_column.get(col, 0) + count
                        file_total_points += count

                if is_no_aggregation:
                    avg_table, removed_rows, column_stats, source_values = no_aggregate_rows(
                        table, selected_rows, selected_columns, outlier_method, outlier_strictness
                    )
                else:
                    avg_table, removed_rows, column_stats, source_values = aggregate_rows(
                        table, selected_rows, selected_columns, agg_func, outlier_method, outlier_strictness
                    )

            if file_time is not None:
                total_points_per_time[file_time] = total_points_per_time.get(file_time, 0) + file_total_points

            aggregated_tables[name] = avg_table
            for col_name, stats in column_stats.items():
                error_lookup[(name, str(col_name))] = stats
            for col_name, vals in source_values.items():
                source_value_lookup[(name, str(col_name))] = vals
            for item in removed_rows:
                removed_outliers_log.append({"Time": file_time, "file": name, **item})
        except Exception:
            continue

    if not aggregated_tables:
        st.error("No aggregated tables could be produced for the current selection")
        return

    combined_rows: list[dict[str, float | str | None]] = []
    for name, avg_table in aggregated_tables.items():
        row = {k: float(v) for k, v in avg_table.iloc[0].to_dict().items()}
        row["Time"] = _extract_time_from_filename(name)
        row["File"] = name
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    combined_df = combined_df.sort_values(by="Time", na_position="last").reset_index(drop=True)
    metric_columns = [c for c in _sort_columns_natural(metric_labels) if c in combined_df.columns]
    display_columns = ["Time", *metric_columns]

    st.header("Aggregated table (all files)")
    combined_display_df = combined_df[display_columns].copy()
    combined_display_df["Time"] = combined_display_df["Time"].map(
        lambda v: _format_time_label(v) if isinstance(v, (float, int, np.floating, np.integer)) else str(v)
    )
    st.dataframe(_gradient_style(combined_display_df, color_scale_mode, color_palette), width="stretch")

    st.header("Statistical analysis")
    stat_col1, stat_col2 = st.columns(2)

    unique_times = sorted(combined_df["Time"].dropna().unique().tolist())
    if not unique_times:
        st.warning("No Time values available for statistical analysis.")
        return

    with stat_col1:
        selected_stat_time = st.selectbox(
            "Row (from Aggregated table)",
            options=unique_times,
            format_func=_format_time_label,
        )

    with stat_col2:
        selected_stat_metric = st.selectbox(
            "Column (from Aggregated table)",
            options=[c for c in _sort_columns_natural(metric_labels) if c in combined_df.columns],
        )

    selected_files = combined_df.loc[combined_df["Time"] == selected_stat_time, "File"].astype(str).tolist()
    stat_values: list[float] = []
    for file_name in selected_files:
        stat_values.extend(source_value_lookup.get((file_name, str(selected_stat_metric)), []))
    stat_table = _basic_stat_metrics(stat_values)
    st.dataframe(stat_table, width="stretch")

    st.subheader("T-test analysis")
    ttest_col1, ttest_col2, ttest_col3, ttest_col4 = st.columns(4)
    with ttest_col1:
        ttest_time_a = st.selectbox("Group A", options=unique_times, format_func=_format_time_label, key="ttest_time_a")
    with ttest_col2:
        ttest_metric_a = st.selectbox("Group A Column", options=metric_columns, key="ttest_metric_a")

    with ttest_col3:
        ttest_time_b = st.selectbox(
            "Group B",
            options=unique_times,
            format_func=_format_time_label,
            key="ttest_time_b",
        )
    with ttest_col4:
        ttest_metric_b = st.selectbox("Group B Column", options=metric_columns, key="ttest_metric_b")

    files_a = combined_df.loc[combined_df["Time"] == ttest_time_a, "File"].astype(str).tolist()
    files_b = combined_df.loc[combined_df["Time"] == ttest_time_b, "File"].astype(str).tolist()
    values_a: list[float] = []
    values_b: list[float] = []
    for file_name in files_a:
        values_a.extend(source_value_lookup.get((file_name, str(ttest_metric_a)), []))
    for file_name in files_b:
        values_b.extend(source_value_lookup.get((file_name, str(ttest_metric_b)), []))

    ttest_df = _welch_t_test(values_a, values_b)
    st.dataframe(ttest_df, width="stretch")

    st.subheader("Pairwise T-test p-values matrix (row-column vs row-column)")
    group_values: dict[str, list[float]] = {}
    ordered_group_labels: list[str] = []
    for tm in unique_times:
        tm_label = _format_time_label(tm)
        files_tm = combined_df.loc[combined_df["Time"] == tm, "File"].astype(str).tolist()
        for metric in metric_columns:
            label = f"{tm_label}-{metric}"
            vals_metric: list[float] = []
            for file_name in files_tm:
                vals_metric.extend(source_value_lookup.get((file_name, str(metric)), []))
            group_values[label] = vals_metric
            ordered_group_labels.append(label)

    pairwise_matrix_df = pd.DataFrame("", index=ordered_group_labels, columns=ordered_group_labels)
    for i, label_i in enumerate(ordered_group_labels):
        for j, label_j in enumerate(ordered_group_labels):
            if i == j:
                pairwise_matrix_df.loc[label_i, label_j] = "1.0"
            elif j > i:
                p_val = _welch_p_value(group_values[label_i], group_values[label_j])
                if pd.isna(p_val):
                    display_p = "N/A"
                elif p_val < 0.005:
                    display_p = "<0.005"
                else:
                    display_p = f"{p_val:.6f}"
                pairwise_matrix_df.loc[label_i, label_j] = display_p
                pairwise_matrix_df.loc[label_j, label_i] = display_p

    if pairwise_matrix_df.empty:
        st.info("Not enough data for pairwise row-column T-tests.")
    else:
        st.dataframe(pairwise_matrix_df, width="stretch")

    removed_df_export = pd.DataFrame(removed_outliers_log)
    if not removed_df_export.empty:
        removed_df_export = removed_df_export.sort_values(by=["Time", "column", "row_id"], na_position="last")

    stats_rows: list[dict[str, float | int | str]] = []
    if outlier_method != "None":
        removed_total = int(len(removed_df_export))
        total_points = int(sum(total_points_per_column.values()))
        overall_pct = (removed_total / total_points * 100) if total_points else 0.0
        stats_rows.append({"Scope": "Overall", "Key": "All", "Total points": total_points, "Removed": removed_total, "Removed %": round(overall_pct, 2)})

        for col in _sort_columns_natural(metric_labels):
            col_total = int(total_points_per_column.get(col, 0))
            col_removed = int((removed_df_export["column"] == col).sum()) if not removed_df_export.empty else 0
            col_pct = (col_removed / col_total * 100) if col_total else 0.0
            stats_rows.append({"Scope": "Column", "Key": str(col), "Total points": col_total, "Removed": col_removed, "Removed %": round(col_pct, 2)})

        for tm in sorted(total_points_per_time.keys()):
            tm_total = int(total_points_per_time[tm])
            tm_removed = int((removed_df_export["Time"] == tm).sum()) if not removed_df_export.empty else 0
            tm_pct = (tm_removed / tm_total * 100) if tm_total else 0.0
            stats_rows.append({"Scope": "Time", "Key": str(tm), "Total points": tm_total, "Removed": tm_removed, "Removed %": round(tm_pct, 2)})

    outlier_stats_df = pd.DataFrame(stats_rows)

    st.header("Outlier filtering log")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Removed values by file/row/column")
        if outlier_method == "None":
            st.info("Outlier filtering is disabled.")
        elif not removed_outliers_log:
            st.info("No outliers were removed with current method/settings.")
        else:
            st.dataframe(removed_df_export, width="stretch")

    with right_col:
        st.subheader("Outlier statistics")
        if outlier_method == "None":
            st.info("Outlier filtering is disabled.")
        else:
            st.dataframe(outlier_stats_df, width="stretch")

    st.header("Cross-file plot")
    if "plot_columns" not in st.session_state:
        st.session_state.plot_columns = list(metric_labels)
    st.session_state.plot_columns = [c for c in st.session_state.plot_columns if c in metric_labels]

    plot_columns = st.multiselect(
        "Columns to display on one chart (different colors)",
        options=metric_columns,
        key="plot_columns",
    )

    controls_col1, controls_col2 = st.columns(2)
    with controls_col1:
        plot_type = st.selectbox(
            "Plot type",
            options=["Line + points", "Line + points (по Columns)", "Scatter", "Bar plot (по Time)", "Bar plot (по Columns)", "Box plot (по Time)", "Box plot (по Columns)", "Violin plot (по Time)", "Violin plot (по Columns)"],
            index=0,
            key="plot_type",
        )
    with controls_col2:
        error_method = st.selectbox(
            "Error bar method",
            options=["None", "Std", "SEM", "95% CI", "IQR/2", "MAD"],
            index=0,
            key="error_method",
        )

    plot_columns = [c for c in plot_columns if c in metric_labels]
    if not plot_columns:
        st.warning("Please choose at least one column for chart")
        return

    plot_df = combined_df[["File", "Time", *[c for c in _sort_columns_natural(metric_labels) if c in combined_df.columns]]].copy()
    available_plot_columns = [c for c in plot_columns if c in plot_df.columns]
    if not available_plot_columns:
        st.warning("No data for selected chart columns")
        return

    chart_long = plot_df[["File", "Time", *available_plot_columns]].melt(
        id_vars=["File", "Time"], var_name="Column", value_name="Value"
    )
    chart_long = chart_long.dropna(subset=["Value", "Time"]).copy()
    sorted_cols = _sort_columns_natural(available_plot_columns)
    chart_long["Column"] = pd.Categorical(chart_long["Column"], categories=sorted_cols, ordered=True)
    chart_long["TimeLabel"] = chart_long["Time"].map(_format_time_label)
    time_sort_order = [_format_time_label(v) for v in sorted(chart_long["Time"].dropna().unique())]

    error_field_map = {
        "Std": "std",
        "SEM": "sem",
        "95% CI": "ci95",
        "IQR/2": "iqr_half",
        "MAD": "mad",
    }
    if error_method != "None":
        field = error_field_map[error_method]
        chart_long["Error"] = chart_long.apply(
            lambda r: float(error_lookup.get((str(r["File"]), str(r["Column"])), {}).get(field, 0.0)), axis=1
        )
    else:
        chart_long["Error"] = 0.0

    chart_long["y_low"] = chart_long["Value"] - chart_long["Error"]
    chart_long["y_high"] = chart_long["Value"] + chart_long["Error"]

    y_candidates = chart_long["Value"]
    if error_method != "None" and plot_type in {"Line + points", "Line + points (по Columns)", "Scatter", "Bar plot (по Time)", "Bar plot (по Columns)"}:
        y_candidates = pd.concat([chart_long["y_low"], chart_long["y_high"]], ignore_index=True)
    y_min = float(y_candidates.min()) if not y_candidates.empty else 0.0
    y_max = float(y_candidates.max()) if not y_candidates.empty else 1.0
    pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=True)

    x_min = float(chart_long["Time"].min()) if not chart_long.empty else 0.0
    x_max = float(chart_long["Time"].max()) if not chart_long.empty else 1.0
    x_pad = (x_max - x_min) * 0.04 if x_max > x_min else 0.5
    x_scale = alt.Scale(domain=[x_min - x_pad, x_max + x_pad], nice=True)
    cap_half_width = max((x_max - x_min) * 0.006, 0.02)
    chart_long["x_left"] = chart_long["Time"] - cap_half_width
    chart_long["x_right"] = chart_long["Time"] + cap_half_width

    if plot_type == "Scatter":
        chart = (
            alt.Chart(chart_long)
            .mark_point(filled=True, size=130, opacity=0.95)
            .encode(
                x=alt.X("Time:Q", title="Time", scale=x_scale),
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
                color=alt.Color("Column:N", title="Column"),
                tooltip=["Time", "File", "Column", "Value"],
            )
            .interactive()
            .properties(height=450)
        )
    elif plot_type == "Bar plot (по Time)":
        chart = (
            alt.Chart(chart_long)
            .mark_bar(opacity=0.88)
            .encode(
                x=alt.X("TimeLabel:N", title="Time", sort=time_sort_order),
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale, stack=None),
                xOffset=alt.XOffset("Column:N"),
                color=alt.Color("Column:N", title="Column"),
                tooltip=["Time", "File", "Column", "Value"],
            )
            .properties(height=450)
        )
    elif plot_type == "Bar plot (по Columns)":
        chart = (
            alt.Chart(chart_long)
            .mark_bar(opacity=0.88)
            .encode(
                x=alt.X("Column:N", title="Column", sort=sorted_cols),
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale, stack=None),
                xOffset=alt.XOffset("TimeLabel:N"),
                color=alt.Color("TimeLabel:N", title="Time"),
                tooltip=["Time", "File", "Column", "Value"],
            )
            .properties(height=450)
        )
    elif plot_type == "Box plot (по Time)":
        chart = (
            alt.Chart(chart_long)
            .mark_boxplot(extent="min-max", size=24)
            .encode(
                x=alt.X("TimeLabel:N", title="Time", sort=time_sort_order),
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
                color=alt.Color("TimeLabel:N", title="Time"),
                tooltip=["Time", "Value"],
            )
            .properties(height=460)
        )
    elif plot_type == "Box plot (по Columns)":
        chart = (
            alt.Chart(chart_long)
            .mark_boxplot(extent="min-max", size=24)
            .encode(
                x=alt.X("Column:N", title="Column", sort=sorted_cols),
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
                color=alt.Color("Column:N", title="Column"),
                tooltip=["Column", "Value"],
            )
            .properties(height=460)
        )
    elif plot_type == "Violin plot (по Time)":
        if chart_long.empty:
            st.warning("No data available for violin plot")
            return

        violin_base = (
            alt.Chart(chart_long)
            .transform_density(
                "Value",
                as_=["Value", "Density"],
                groupby=["TimeLabel"],
            )
            .mark_area(orient="horizontal", opacity=0.62, color="#ff6ab6")
            .encode(
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
                x=alt.X("Density:Q", stack="center", axis=None),
            )
        )
        chart = violin_base.facet(
            column=alt.Column("TimeLabel:N", title="Time", sort=time_sort_order, header=alt.Header(labelAngle=0)),
            columns=4,
        ).properties(height=180, width=80)
    elif plot_type == "Violin plot (по Columns)":
        chart = (
            alt.Chart(chart_long)
            .transform_density(
                "Value",
                as_=["Value", "Density"],
                groupby=["Column"],
            )
            .mark_area(orient="horizontal", opacity=0.62)
            .encode(
                y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
                x=alt.X("Density:Q", stack="center", axis=None),
                color=alt.Color("Column:N", title="Column"),
                column=alt.Column("Column:N", title="Column", sort=sorted_cols, spacing=10),
            )
            .properties(height=440)
        )
    elif plot_type == "Line + points (по Columns)":
        by_col = chart_long.copy()
        by_col["Column"] = by_col["Column"].astype(str)

        base = alt.Chart(by_col).encode(
            x=alt.X("Column:N", title="Column", sort=sorted_cols),
            y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
            color=alt.Color("TimeLabel:N", title="Time"),
            detail=alt.Detail("File:N"),
            tooltip=["TimeLabel", "File", "Column", "Value", "Error"],
        )

        main_chart = base.mark_line(strokeWidth=3.0, point=alt.OverlayMarkDef(size=95, filled=True))
        use_error = error_method != "None"

        if use_error:
            error_chart = alt.Chart(by_col).mark_rule(opacity=0.55, strokeWidth=1.6).encode(
                x=alt.X("Column:N", sort=sorted_cols),
                y=alt.Y("y_low:Q"),
                y2=alt.Y2("y_high:Q"),
                color=alt.Color("TimeLabel:N"),
                detail=alt.Detail("File:N"),
            )
            chart = (error_chart + main_chart).interactive().properties(height=450)
        else:
            chart = main_chart.interactive().properties(height=450)
    else:
        base = alt.Chart(chart_long).encode(
            x=alt.X("Time:Q", title="Time", scale=x_scale),
            y=alt.Y("Value:Q", title="Aggregated value", scale=y_scale),
            color=alt.Color("Column:N", title="Column"),
            tooltip=["Time", "File", "Column", "Value", "Error"],
        )

        main_chart = base.mark_line(strokeWidth=3.0, point=alt.OverlayMarkDef(size=95, filled=True))
        use_error = error_method != "None"

        if use_error:
            error_chart = alt.Chart(chart_long).mark_rule(opacity=0.55, strokeWidth=1.6).encode(
                x=alt.X("Time:Q", scale=x_scale),
                y=alt.Y("y_low:Q"),
                y2=alt.Y2("y_high:Q"),
                color=alt.Color("Column:N"),
            )
            whisker_low = alt.Chart(chart_long).mark_rule(opacity=0.7, strokeWidth=1.4).encode(
                x=alt.X("x_left:Q", scale=x_scale),
                x2=alt.X2("x_right:Q"),
                y=alt.Y("y_low:Q"),
                color=alt.Color("Column:N"),
            )
            whisker_high = alt.Chart(chart_long).mark_rule(opacity=0.7, strokeWidth=1.4).encode(
                x=alt.X("x_left:Q", scale=x_scale),
                x2=alt.X2("x_right:Q"),
                y=alt.Y("y_high:Q"),
                color=alt.Color("Column:N"),
            )
            chart = (error_chart + whisker_low + whisker_high + main_chart).interactive().properties(height=450)
        else:
            chart = main_chart.interactive().properties(height=450)

    st.altair_chart(chart, use_container_width=True)
    plot_display_df = plot_df[["Time", *available_plot_columns]].copy()
    plot_display_df["Time"] = plot_display_df["Time"].map(_format_time_label)
    st.dataframe(_gradient_style(plot_display_df, color_scale_mode, color_palette), width="stretch")

    st.subheader("Export")
    report_sheets = {
        "Aggregated table": combined_display_df,
        "Statistical analysis": stat_table,
        "T-test analysis": ttest_df,
        "Pairwise T-test p-values": pairwise_matrix_df,
        "Outlier log": removed_df_export,
        "Outlier statistics": outlier_stats_df,
        "Plot data": plot_display_df,
    }
    report_bytes = _build_excel_report(report_sheets)
    st.download_button(
        "Export report.xlsx",
        data=report_bytes,
        file_name="plate_master_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )



def main() -> None:
    if get_script_run_ctx is not None and get_script_run_ctx() is None:
        try:
            os.execvp("streamlit", ["streamlit", "run", str(Path(__file__).resolve())])
        except FileNotFoundError:
            print("Streamlit is not installed in PATH. Run: pip install -r requirements.txt")
        return
    run_dashboard()


if __name__ == "__main__":
    main()
