import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results/backtests"
LATEX_FILE = BASE_DIR / "txt/3rd_part.tex"
MAIN_LATEX_FILE = BASE_DIR / "txt/main.tex"

def load_data():
    summary = pd.read_csv(RESULTS_DIR / "backtest_summary.csv")
    detailed = pd.read_csv(RESULTS_DIR / "backtest_summary_detailed.csv")
    family = pd.read_csv(RESULTS_DIR / "model_family_summary.csv")
    ablation = pd.read_csv(RESULTS_DIR / "google_trends_ablation_summary.csv")
    bias = pd.read_csv(RESULTS_DIR / "residual_bias_summary.csv")
    predictions = pd.read_csv(RESULTS_DIR / "backtest_predictions.csv")
    return {
        'summary': summary,
        'detailed': detailed,
        'family': family,
        'ablation': ablation,
        'bias': bias,
        'predictions': predictions
    }

def update_latex(content, data):
    # 1. Update in-text MAPE (lines 369-371)
    for stage in ['Early', 'Mid', 'Late']:
        val = data['summary'][(data['summary']['model'] == 'StackingNowcast') & (data['summary']['stage'] == stage)]['mape'].values[0]
        content = re.sub(rf"\\textbf{{{stage}}}: \\texttt{{StackingNowcast}}, MAPE \$= [\d\.]+\%\$;", 
                        f"\\textbf{{{stage}}}: \\texttt{{StackingNowcast}}, MAPE $= {val:.3f}\%$;", content)

    # 2. Update Table: headline_accuracy
    # Pattern: Model & Early & Mid & Late \\
    for _, row in data['summary'].groupby('model'):
        model = row['model'].values[0]
        early = row[row['stage'] == 'Early']['mape'].values[0] if 'Early' in row['stage'].values else None
        mid = row[row['stage'] == 'Mid']['mape'].values[0] if 'Mid' in row['stage'].values else None
        late = row[row['stage'] == 'Late']['mape'].values[0] if 'Late' in row['stage'].values else None
        
        # Replacement pattern
        if early is not None and mid is not None and late is not None:
            # For 3-column stage tables
            pattern = rf"({model}\s*&\s*)[\d\.]+(\s*&\s*)[\d\.]+(\s*&\s*)[\d\.]+"
            content = re.sub(pattern, rf"\g<1>{early:.3f}\g<2>{mid:.3f}\g<3>{late:.3f}", content)
        elif model == 'EarlyShockAdjusted' and early is not None:
             pattern = rf"({model}\s*&\s*)[\d\.]+(\s*&\s*--\s*&\s*--)"
             content = re.sub(pattern, rf"\g<1>{early:.3f}\g<2>", content)

    # 3. Update Table: google_ablation (Base, Google, Market, Full)
    # The ablation summary CSV might have different model names like 'base', 'base_google', etc.
    # I'll check the ablation summary content.
    
    # 4. Update Table: top5_by_stage
    # This one is tricky because it's ranked. I'll skip auto-ranking for now and just update if they match.
    
    # 5. Update Table: family_stage_summary
    # Pivot the family summary to get stages as columns
    fam_avg = data['family'].groupby(['family', 'stage'])['mape'].mean().unstack()
    for fam, row in fam_avg.iterrows():
        early = row.get('Early', None)
        mid = row.get('Mid', None)
        late = row.get('Late', None)
        if early is not None and mid is not None and late is not None:
            pattern = rf"({fam}\s*&\s*)[\d\.]+(\s*&\s*)[\d\.]+(\s*&\s*)[\d\.]+"
            content = re.sub(pattern, rf"\g<1>{early:.3f}\g<2>{mid:.3f}\g<3>{late:.3f}", content)

    # 6. Update Table: robustness_selected (MAPE, Shock MAPE, Non-shock MAPE, Bias)
    # This table has: Model & Stage & Overall MAPE & Shock MAPE & Non-shock MAPE & Prediction bias \\
    # I'll check bias CSV.
    
    # 7. Update Table: factor_comparison (AR, Bridge, MIDAS, DFM, DFM-SA)
    # Already handled by the general loop in step 2.

    # 8. Update Table: shock_quarter_results (APE for specific quarters)
    # 2020 Q2 & Early & 15.827 & 18.045 & 13.854 & 19.964 & 11.459 \\
    # Models: AdaEns, ENet, Bridge, DFM-SA, ESA
    # I'll check predictions for 2020-04-01 (2020 Q2)
    q2_2020 = data['predictions'][data['predictions']['prediction_date'] == '2020-04-01']
    if not q2_2020.empty:
        # Map abbreviations to model names
        model_map = {'AdaEns': 'AdaptiveEnsemble', 'ENet': 'ElasticNet', 'Bridge': 'Bridge', 'DFM-SA': 'DFMShockAdjusted', 'ESA': 'EarlyShockAdjusted'}
        def get_ape(model, stage):
            m = q2_2020[(q2_2020['model'] == model_map[model]) & (q2_2020['stage'] == stage)]
            return m['abs_pct_error'].values[0] if not m.empty else None
        
        # 2020 Q2 Early
        vals = [get_ape(m, 'Early') for m in ['AdaEns', 'ENet', 'Bridge', 'DFM-SA', 'ESA']]
        if all(v is not None for v in vals):
            pattern = r"(2020 Q2\s*&\s*Early\s*&\s*)[\d\.]+\s*&\s*[\d\.]+\s*&\s*[\d\.]+\s*&\s*[\d\.]+\s*&\s*[\d\.]+"
            content = re.sub(pattern, rf"\g<1>{vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.3f}", content)

    return content

if __name__ == "__main__":
    data = load_data()
    
    # Update 3rd_part.tex
    with open(LATEX_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = update_latex(content, data)
    with open(LATEX_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Updated 3rd_part.tex")
    
    # Update main.tex
    with open(MAIN_LATEX_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = update_latex(content, data)
    with open(MAIN_LATEX_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Updated main.tex")
