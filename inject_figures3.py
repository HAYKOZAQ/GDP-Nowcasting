#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for idx, line in enumerate(lines):
    if '["Գլխավոր էջ", "Մոդելների վերլուծություն"],' in line:
        line = line.replace('["Գլխավոր էջ", "Մոդելների վերլուծություն"],', '["Գլխավոր էջ", "Մոդելների վերլուծություն", "Գծապատկերներ"],')
    new_lines.append(line)
    
    # Check if we are at line 295 which is exactly '        st.stop()\n' inside the nowcasting main page block
    # We can ensure it by checking the previous few lines to see if it's the forecast_chart plotly plotting.
    if line == '        st.stop()\n' and 'plotly_chart' in lines[idx-2]:
        figures_block = """
    elif nowcast_section == "Գծապատկերներ":
        import os as _os
        _FIG_DIR = _os.path.join(BASE_DIR, "nowcasting_results", "figures")
        _FALLBACK = _os.path.join(_os.path.dirname(BASE_DIR), "nowcasting-CODE", "results", "figures")
        _fig_dir = _FIG_DIR if _os.path.isdir(_FIG_DIR) else (_FALLBACK if _os.path.isdir(_FALLBACK) else None)
        st.markdown(
            "<div style='background:rgba(31,111,235,0.13);border-left:4px solid #58a6ff;"
            "padding:12px 18px;border-radius:10px;margin:0 0 18px 0;'>"
            "<strong>Nowcast Գծապատկերների Արխիվ.</strong> "
            "Բոլոր 15 գծապատկերները ավտոմատ կերպով ստեղծվել են վերջին walk-forward backtest-ի (StackingNowcast) արդյունքում:</div>",
            unsafe_allow_html=True,
        )
        _figures = [
            ("backtest_stage_mape.png",               "Մոդելների ճշգրտությունն ըստ փուլի (MAPE %)"),
            ("stage_winner_small_multiples.png",       "Լավագույն մոդելն ըստ փուլի — Փաստացի vs Կանխատեսված"),
            ("backtest_best_late_model.png",           "Ուշ փուլի լավագույն մոդել — StackingNowcast"),
            ("backtest_dfm_shock_adjusted.png",        "DFMShockAdjusted ճգնաժամային benchmark"),
            ("adaptive_ensemble_spaghetti.png",        "Ensemble spaghetti — բոլոր մոդելների կանխատեսման ուղեգծերը"),
            ("forecast_uncertainty_bands.png",         "Կանխատեսման անորոշության միջակայքեր"),
            ("focus_2020q2_models.png",                "2020-Q2 (COVID) ճնշում — մոդելների համեմատություն"),
            ("focus_2020q2_information_set.png",       "2020-Q2 — Տեղեկատվական փաթեթի ընդլայնում ըստ փուլի"),
            ("release_calendar_information_flow.png",  "Տեղեկատվական հոսքի ժամանակացույց ըստ փուլի"),
            ("model_ranking_heatmap.png",              "Մոդելների վարկանիշային heatmap ըստ փուլի"),
            ("model_family_comparison.png",            "Մոդելների ընտանիքների համեմատություն — ML vs Structural vs Combination"),
            ("family_shock_nonshock.png",              "Ճնշումային vs սովորական եռամսյակ — ճշգրտություն ըստ ընտանիքի"),
            ("selected_model_bias_profile.png",        "Ընտրված մոդելների bias profile ըստ փուլի"),
            ("google_trends_marginal_value.png",       "Google Trends-ի լրացուցիչ արժեքը (ablation)"),
            ("future_gdp_forecast_dark.png",           "2026թ. ՀՆԱ-ի կանխատեսումն ըստ ռեկուրսիվ մոդելի"),
        ]
        if _fig_dir is None:
            st.error("Figure directory not found.")
        else:
            for _i in range(0, len(_figures), 2):
                _cols = st.columns(2)
                for _j, _col in enumerate(_cols):
                    if _i + _j >= len(_figures):
                        break
                    _fname, _caption = _figures[_i + _j]
                    _fpath = _os.path.join(_fig_dir, _fname)
                    if _os.path.exists(_fpath):
                        _col.image(_fpath, caption=_caption, use_container_width=True)
                    else:
                        _col.warning(f"{_fname} not found")
        st.stop()
"""
        new_lines.append(figures_block)

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

import ast
try:
    with open("dashboard.py", "r", encoding="utf-8") as f:
        ast.parse(f.read())
    print("Dashboard patched successfully and syntax is OK.")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
