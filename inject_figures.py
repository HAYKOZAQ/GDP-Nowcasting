#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patches dashboard.py to add a Figures tab. Diagnoses syntax error."""
import re, shutil, ast, sys

src = "dashboard.py"
bak = "dashboard.py.bak"
shutil.copy(src, bak)

with open(src, "r", encoding="utf-8") as f:
    txt = f.read()

# ── 1. Add third radio option ────────────────────────────────────────────────
radio_re = re.compile(
    r'(\["[^"]+", "[^"]+"\],\s*\n\s*horizontal=True,\s*\n\s*key="nowcast_section",\s*\n\s*\))',
    re.DOTALL,
)
m = radio_re.search(txt)
if m:
    old = m.group(0)
    if "Nowcast Figures" not in old:
        new = old.replace("],", ', "Nowcast Figures"],', 1)
        txt = txt.replace(old, new, 1)
        sys.stdout.buffer.write(b"Radio option added.\n")
    else:
        sys.stdout.buffer.write(b"Radio already patched.\n")
else:
    sys.stdout.buffer.write(b"WARNING: radio not found\n")

# ── 2. Build the block as a list of lines (no triple-quote issues) ────────────
block_lines = [
    "",
    "    elif nowcast_section == \"Nowcast Figures\":",
    "        import os as _os",
    "        _FIG_DIR = _os.path.join(BASE_DIR, \"nowcasting_results\", \"figures\")",
    "        _FALLBACK = _os.path.join(_os.path.dirname(BASE_DIR), \"nowcasting-CODE\", \"results\", \"figures\")",
    "        _fig_dir = _FIG_DIR if _os.path.isdir(_FIG_DIR) else (_FALLBACK if _os.path.isdir(_FALLBACK) else None)",
    "        st.markdown(",
    "            \"<div style='background:rgba(31,111,235,0.13);border-left:4px solid #58a6ff;\"",
    "            \"padding:12px 18px;border-radius:10px;margin:0 0 18px 0;'>\"",
    "            \"<strong>Nowcast Figures Archive.</strong> All 15 charts auto-generated from the \"",
    "            \"latest walk-forward backtest (StackingNowcast era). Updated 2026-04-23.</div>\",",
    "            unsafe_allow_html=True,",
    "        )",
    "        _figures = [",
    "            (\"backtest_stage_mape.png\",               \"Model MAPE by Information Stage\"),",
    "            (\"stage_winner_small_multiples.png\",       \"Best Model per Stage vs Actual GDP\"),",
    "            (\"backtest_best_late_model.png\",           \"Late Stage Winner: StackingNowcast\"),",
    "            (\"backtest_dfm_shock_adjusted.png\",        \"DFMShockAdjusted - Crisis Benchmark\"),",
    "            (\"adaptive_ensemble_spaghetti.png\",        \"Ensemble Spaghetti - All Model Paths\"),",
    "            (\"forecast_uncertainty_bands.png\",         \"Forecast Uncertainty Bands (50% & 90%)\"),",
    "            (\"focus_2020q2_models.png\",                \"2020-Q2 COVID Quarter - Model Comparison\"),",
    "            (\"focus_2020q2_information_set.png\",       \"2020-Q2 Information Set by Stage\"),",
    "            (\"release_calendar_information_flow.png\",  \"Data Release Calendar & Information Flow\"),",
    "            (\"model_ranking_heatmap.png\",              \"Model Ranking Heatmap by Stage\"),",
    "            (\"model_family_comparison.png\",            \"Model Family: ML vs Structural vs Combination\"),",
    "            (\"family_shock_nonshock.png\",              \"Shock vs Non-Shock MAPE by Model Family\"),",
    "            (\"selected_model_bias_profile.png\",        \"Bias Profile of Selected Models by Stage\"),",
    "            (\"google_trends_marginal_value.png\",       \"Google Trends Marginal Value (Ablation)\"),",
    "            (\"future_gdp_forecast_dark.png\",           \"Armenia GDP Forecast 2026 Q2-Q4\"),",
    "        ]",
    "        if _fig_dir is None:",
    "            st.error(\"Figure directory not found.\")",
    "        else:",
    "            for _i in range(0, len(_figures), 2):",
    "                _cols = st.columns(2)",
    "                for _j, _col in enumerate(_cols):",
    "                    if _i + _j >= len(_figures):",
    "                        break",
    "                    _fname, _caption = _figures[_i + _j]",
    "                    _fpath = _os.path.join(_fig_dir, _fname)",
    "                    if _os.path.exists(_fpath):",
    "                        _col.image(_fpath, caption=_caption, use_container_width=True)",
    "                    else:",
    "                        _col.warning(f\"{_fname} not found\")",
    "",
]
BLOCK = "\n".join(block_lines) + "\n"

# ── 3. Insert before next elif page == ───────────────────────────────────────
marker = '            st.dataframe(latest_snapshot, width="stretch", hide_index=True)\n\nelif page =='
idx = txt.find(marker)
if idx >= 0:
    insert_at = idx + len('            st.dataframe(latest_snapshot, width="stretch", hide_index=True)\n')
    txt = txt[:insert_at] + BLOCK + txt[insert_at:]
    sys.stdout.buffer.write(b"Figures block inserted.\n")
else:
    sys.stdout.buffer.write(b"WARNING: marker not found\n")

with open(src, "w", encoding="utf-8") as f:
    f.write(txt)

# ── 4. Verify syntax ──────────────────────────────────────────────────────────
try:
    ast.parse(txt)
    sys.stdout.buffer.write(b"Syntax OK.\n")
except SyntaxError as e:
    msg = f"Syntax ERROR at line {e.lineno}: {e.msg}\n".encode()
    sys.stdout.buffer.write(msg)
    # Show context
    lines = txt.splitlines()
    for li in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
        out = f"  {li+1}: {lines[li]}\n".encode("utf-8", errors="replace")
        sys.stdout.buffer.write(out)
    shutil.copy(bak, src)
    sys.stdout.buffer.write(b"Backup restored.\n")
