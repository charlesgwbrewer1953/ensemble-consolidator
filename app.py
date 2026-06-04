import math
import io
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
MISSING_SENTINEL = -3

SHEET_ALIASES = {
    "LB":         "LAB",
    "CGPT_SNP":   "SNP",
    "GEMINI_SNP": "SNP",
}

FUNCTION_LABELS = {
    "null":      "Null — plain average (identity)",
    "squared":   "Squared  x²",
    "cubed":     "Cubed  x³",
    "lonn":      "Natural log  ln(x)",
    "cos":       "Cosine  cos(x)",
    "logistic":  "Logistic  p = 1 / (1 + e^(−kx))",
}

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def normalise_sheet(name: str) -> str:
    upper = name.strip().upper()
    return SHEET_ALIASES.get(upper, upper)


def make_transform(name: str, k: float = 1.0):
    if name == "null":
        return lambda x: x
    if name == "squared":
        return lambda x: x ** 2
    if name == "cubed":
        return lambda x: x ** 3
    if name == "lonn":
        return lambda x: math.log(x) if x > 0 else float("nan")
    if name == "cos":
        return lambda x: math.cos(x)
    if name == "logistic":
        def logistic(x):
            try:
                return 1.0 / (1.0 + math.exp(-k * x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0
        return logistic
    raise ValueError(f"Unknown function: {name}")


def load_from_paths(file_paths: list[str]) -> dict[str, list[pd.DataFrame]]:
    """Load xlsx files given a list of filesystem paths."""
    all_data: dict[str, list[pd.DataFrame]] = {}
    for path in file_paths:
        filename = os.path.basename(path)
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            st.warning(f"Could not open {filename}: {e}")
            continue
        for sheet in xl.sheet_names:
            key = normalise_sheet(sheet)
            if key not in all_data:
                all_data[key] = []
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=None)
                all_data[key].append(df)
            except Exception as e:
                st.warning(f"Skipped '{sheet}' in {filename}: {e}")
    return all_data


def load_from_uploads(uploaded_files) -> dict[str, list[pd.DataFrame]]:
    """Load xlsx files from Streamlit UploadedFile objects."""
    all_data: dict[str, list[pd.DataFrame]] = {}
    for uf in uploaded_files:
        xl = pd.ExcelFile(uf)
        for sheet in xl.sheet_names:
            key = normalise_sheet(sheet)
            if key not in all_data:
                all_data[key] = []
            try:
                df = pd.read_excel(uf, sheet_name=sheet, header=None)
                all_data[key].append(df)
            except Exception as e:
                st.warning(f"Skipped '{sheet}' in {uf.name}: {e}")
    return all_data


def average_sheets(dfs: list[pd.DataFrame]):
    max_rows = max(df.shape[0] for df in dfs)
    max_cols = max(df.shape[1] for df in dfs)

    avg_df = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)
    lbl_df = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)

    for r in range(max_rows):
        for c in range(max_cols):
            nums, label = [], None
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
                avg_df.iat[r, c] = float(np.mean(nums))
            if label is not None:
                lbl_df.iat[r, c] = label

    return avg_df, lbl_df


def apply_transform(avg_df: pd.DataFrame, lbl_df: pd.DataFrame, fn) -> pd.DataFrame:
    max_rows, max_cols = avg_df.shape
    result = pd.DataFrame(np.nan, index=range(max_rows), columns=range(max_cols), dtype=object)

    for r in range(max_rows):
        for c in range(max_cols):
            v   = avg_df.iat[r, c]
            lbl = lbl_df.iat[r, c]
            if isinstance(v, float) and not math.isnan(v):
                result.iat[r, c] = v if v == MISSING_SENTINEL else fn(v)
            elif isinstance(lbl, str):
                result.iat[r, c] = lbl

    return result


def build_xlsx(averaged: dict, fn, all_data: dict) -> tuple[io.BytesIO, list[dict]]:
    buf = io.BytesIO()
    summary_rows = []

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for tab, (avg_df, lbl_df) in averaged.items():
            result_df = apply_transform(avg_df, lbl_df, fn)

            numeric_vals = [
                result_df.iat[r, c]
                for r in range(result_df.shape[0])
                for c in range(result_df.shape[1])
                if isinstance(result_df.iat[r, c], float)
                and not math.isnan(result_df.iat[r, c])
                and result_df.iat[r, c] != MISSING_SENTINEL
            ]

            summary_rows.append({
                "Sheet":   tab,
                "Sources": len(all_data.get(tab, [])),
                "N Cells": len(numeric_vals),
                "Min":     round(min(numeric_vals),            4) if numeric_vals else None,
                "Max":     round(max(numeric_vals),            4) if numeric_vals else None,
                "Mean":    round(float(np.mean(numeric_vals)), 4) if numeric_vals else None,
            })
            result_df.to_excel(writer, sheet_name=tab[:31], index=False, header=False)

    buf.seek(0)
    return buf, summary_rows


APP_VERSION = "v 1.7.0"

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Ensemble Consolidator", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300&display=swap');

.dk-brand {
    font-family: 'Roboto', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    letter-spacing: 0.12em;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.app-version {
    font-family: 'Roboto', sans-serif;
    font-weight: 300;
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: -0.9rem;
    margin-bottom: 1.5rem;
}
</style>
<p class="dk-brand">demographiKon</p>
""", unsafe_allow_html=True)

st.title("📊 AI Ensemble Consolidator")
st.markdown(f'<p class="app-version">{APP_VERSION}</p>', unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    selected_fn = st.selectbox(
        "Transform function",
        list(FUNCTION_LABELS.keys()),
        format_func=lambda x: FUNCTION_LABELS[x],
    )

    k = 1.0
    if selected_fn == "logistic":
        st.markdown("### Logistic parameter")
        k = st.slider(
            "k",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Steepness of the sigmoid curve. Higher k = steeper.",
        )
        st.latex(r"p = \frac{1}{1 + e^{-kx}},\quad k=" + f"{k:.1f}")

    output_stem = (
        f"Ensemble_Output_logistic_k{k:.1f}"
        if selected_fn == "logistic"
        else f"Ensemble_Output_{selected_fn}"
    )

    st.markdown("---")
    st.markdown(f"**Output filename**  \n`{output_stem}.xlsx`")

# ─── SOURCE SELECTION ──────────────────────────────────────────────────────────
source_mode = st.radio(
    "Source files",
    ["Upload files", "Local folder path"],
    horizontal=True,
    help="Use 'Local folder path' when running on this machine to avoid browser upload limits.",
)

all_data: dict | None = None
file_labels: list[str] = []

if source_mode == "Upload files":
    uploaded_files = st.file_uploader(
        "Select .xlsx files",
        type="xlsx",
        accept_multiple_files=True,
    )
    if uploaded_files:
        file_labels = [f.name for f in uploaded_files]
        all_data = load_from_uploads(uploaded_files)

else:  # Local folder path
    default_path = str(os.path.expanduser("~/Desktop/Ensemble/"))
    folder_path = st.text_input(
        "Folder path (all .xlsx files in this folder will be used)",
        value=default_path,
        placeholder="/Users/you/path/to/folder",
    )
    folder_path = os.path.expanduser(folder_path.strip())

    if folder_path:
        if not os.path.isdir(folder_path):
            st.error(f"Folder not found: `{folder_path}`")
        else:
            candidates = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
            paths = [
                p for p in candidates
                if not os.path.basename(p).startswith("~$")
                and not os.path.basename(p).startswith("Ensemble_Output")
            ]
            if not paths:
                st.warning("No eligible .xlsx files found in that folder.")
            else:
                file_labels = [os.path.basename(p) for p in paths]
                st.info(f"Found {len(paths)} file(s): {', '.join(file_labels)}")
                all_data = load_from_paths(paths)

# ─── GENERATE ──────────────────────────────────────────────────────────────────
if all_data is None:
    st.stop()

st.success(f"{len(file_labels)} file(s) loaded — {len(all_data)} unique sheet(s): {', '.join(sorted(all_data))}")

if st.button("Generate", type="primary"):
    with st.spinner("Averaging source files…"):
        averaged = {tab: average_sheets(dfs) for tab, dfs in sorted(all_data.items())}

    with st.spinner("Applying transform and writing output…"):
        fn = make_transform(selected_fn, k)
        xlsx_buf, summary_rows = build_xlsx(averaged, fn, all_data)

    st.success("Done!")

    st.download_button(
        label=f"⬇️  Download  {output_stem}.xlsx",
        data=xlsx_buf,
        file_name=f"{output_stem}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    st.subheader("Output summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
