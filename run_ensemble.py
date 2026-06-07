#!/usr/bin/env python3
"""
Ensemble spreadsheet consolidator.
Reads all numbered .xlsx files, averages numeric values by sheet/tab,
applies every transform in turn, and writes one output file per transform.
"""

import os
import glob
import math
import numpy as np
import pandas as pd

MISSING_SENTINEL = -3  # cells with this value are treated as missing; not transformed

EXPONENTIAL_Y    = 2.0   # base for exponential transform y^x; change as needed
TRANSFORM_WEIGHT = True  # set False to leave Weight column (col 1) untransformed


def _exp_transform(x):
    try:
        result = EXPONENTIAL_Y ** x
        return float("nan") if isinstance(result, complex) else result
    except (ZeroDivisionError, OverflowError):
        return float("nan")


TRANSFORMS = {
    "null":        lambda x: x,
    "squared":     lambda x: x ** 2,
    "cubed":       lambda x: x ** 3,
    "lonn":        lambda x: math.log(x) if x > 0 else float("nan"),
    "cos":         lambda x: math.cos(x),
    "exponential": _exp_transform,
}

# Normalise variant sheet names to a single canonical key
SHEET_ALIASES = {
    "LB":         "LAB",
    "CGPT_SNP":   "SNP",
    "GEMINI_SNP": "SNP",
}

def normalise_sheet(name: str) -> str:
    upper = name.strip().upper()
    return SHEET_ALIASES.get(upper, upper)

# ─── FILE DISCOVERY ────────────────────────────────────────────────────────────
path = os.path.expanduser("~/Desktop/Ensemble/")
all_files = glob.glob(os.path.join(path, "*.xlsx"))
files = sorted([
    f for f in all_files
    if not os.path.basename(f).startswith("~$")
    and not os.path.basename(f).startswith("Ensemble_Output_")
    and os.path.basename(f) != "Ensemble_Output.xlsx"
])

if not files:
    print(f"ERROR: No .xlsx files found in {path}")
    exit(1)

print(f"Source files: {len(files)}")
for f in files:
    print(f"  {os.path.basename(f)}")

# ─── LOAD ──────────────────────────────────────────────────────────────────────
all_data: dict[str, list[pd.DataFrame]] = {}

for f in files:
    filename = os.path.basename(f)
    xl = pd.ExcelFile(f)
    for sheet in xl.sheet_names:
        key = normalise_sheet(sheet)
        if key not in all_data:
            all_data[key] = []
        try:
            df = pd.read_excel(f, sheet_name=sheet, header=None)
            all_data[key].append(df)
        except Exception as e:
            print(f"  Skipping '{sheet}' in {filename}: {e}")

# ─── CONSOLIDATE (average only, no transform) ──────────────────────────────────
def average_sheets(dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (averaged_numeric_df, label_df) where:
    - averaged_numeric_df: float cells are averaged; non-numeric cells are NaN
    - label_df: text labels (from first file); numeric cells are NaN
    """
    max_rows = max(df.shape[0] for df in dfs)
    max_cols = max(df.shape[1] for df in dfs)

    avg_result  = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)
    label_result = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)

    for r in range(max_rows):
        for c in range(max_cols):
            nums = []
            label = None
            for df in dfs:
                if r < df.shape[0] and c < df.shape[1]:
                    v = df.iloc[r, c]
                    if pd.isna(v):
                        continue
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        nums.append(float(v))
                    elif isinstance(v, str) and v.strip():
                        label = v

            if nums:
                avg_result.iat[r, c] = float(np.mean(nums))
            if label is not None:
                label_result.iat[r, c] = label

    return avg_result, label_result

def apply_transform(avg_df: pd.DataFrame, label_df: pd.DataFrame, fn, transform_weight: bool = True) -> pd.DataFrame:
    """Build output DataFrame: transform numeric cells, restore text labels."""
    max_rows, max_cols = avg_df.shape
    result = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)

    for r in range(max_rows):
        for c in range(max_cols):
            v = avg_df.iat[r, c]
            lbl = label_df.iat[r, c]
            if isinstance(v, float) and not math.isnan(v):
                if v == MISSING_SENTINEL:
                    result.iat[r, c] = v
                elif c == 1 and not transform_weight:
                    result.iat[r, c] = v
                else:
                    result.iat[r, c] = fn(v)
            elif isinstance(lbl, str):
                result.iat[r, c] = lbl

    return result

# ─── PRE-COMPUTE AVERAGES ──────────────────────────────────────────────────────
print(f"\nAveraging {len(all_data)} sheet(s) across {len(files)} files...")
averaged: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
for tab, dfs in sorted(all_data.items()):
    averaged[tab] = average_sheets(dfs)
    print(f"  {tab}")

# ─── RUN ALL TRANSFORMS ────────────────────────────────────────────────────────
all_summaries = []

for transform_name, fn in TRANSFORMS.items():
    if transform_name == "exponential":
        stem = f"Ensemble_Output_exponential_y{EXPONENTIAL_Y:.1f}"
    else:
        stem = f"Ensemble_Output_{transform_name}"
    output_path = os.path.join(path, f"{stem}.xlsx")
    summary: list[dict] = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for tab, (avg_df, label_df) in averaged.items():
            result_df = apply_transform(avg_df, label_df, fn, TRANSFORM_WEIGHT)

            numeric_vals = [
                result_df.iat[r, c]
                for r in range(result_df.shape[0])
                for c in range(result_df.shape[1])
                if isinstance(result_df.iat[r, c], float)
                and not math.isnan(result_df.iat[r, c])
                and result_df.iat[r, c] != MISSING_SENTINEL
            ]

            summary.append({
                "sheet": tab,
                "n":     len(numeric_vals),
                "min":   round(min(numeric_vals),  4) if numeric_vals else None,
                "max":   round(max(numeric_vals),  4) if numeric_vals else None,
                "mean":  round(float(np.mean(numeric_vals)), 4) if numeric_vals else None,
            })
            result_df.to_excel(writer, sheet_name=tab[:31], index=False, header=False)

    all_summaries.append((transform_name, output_path, summary))
    print(f"\nSaved: {stem}.xlsx")

# ─── EXAMINE OUTPUT ────────────────────────────────────────────────────────────
print()
for transform_name, output_path, summary in all_summaries:
    print(f"{'─'*62}")
    print(f"Transform: {transform_name}")
    print(f"{'Sheet':<15} {'N Cells':>8} {'Min':>10} {'Max':>10} {'Mean':>10}")
    for row in summary:
        print(
            f"{row['sheet']:<15} {row['n']:>8} "
            f"{str(row['min']):>10} {str(row['max']):>10} {str(row['mean']):>10}"
        )
