import pandas as pd
import numpy as np
import pytest

def add_qoq(df):
    """
    Function extracted from nowcast_walkforward.py calculation
    """
    chg = df.pct_change().replace([np.inf, -np.inf], np.nan) * 100
    chg.columns = [f'{c}_QoQ' for c in df.columns]
    return pd.concat([df, chg], axis=1)

def test_add_qoq():
    """
    Test Quarter-over-Quarter calculation
    """
    # Create sample data
    data = {'GT_Term1': [100, 110, 121], 'GT_Term2': [50, 25, 50]}
    df = pd.DataFrame(data, index=pd.date_range(start='2020-01-01', periods=3, freq='QS'))
    
    result = add_qoq(df)
    
    # Check shape
    assert result.shape == (3, 4)
    assert 'GT_Term1_QoQ' in result.columns
    assert 'GT_Term2_QoQ' in result.columns
    
    # Check values
    assert np.isnan(result['GT_Term1_QoQ'].iloc[0])
    assert result['GT_Term1_QoQ'].iloc[1] == pytest.approx(10.0) # (110-100)/100 * 100
    assert result['GT_Term1_QoQ'].iloc[2] == pytest.approx(10.0) # (121-110)/110 * 100
    
    assert result['GT_Term2_QoQ'].iloc[1] == pytest.approx(-50.0) # (25-50)/50 * 100
    assert result['GT_Term2_QoQ'].iloc[2] == pytest.approx(100.0) # (50-25)/25 * 100


def test_almon_weights():
    """
    Test Almon polynomial weighting logic
    """
    k = np.arange(1, 4, dtype=float)
    almon_w = np.exp(-(k-2)**2)
    almon_w /= almon_w.sum()
    
    assert len(almon_w) == 3
    assert np.isclose(almon_w.sum(), 1.0)
    
    # The middle weight (k=2) should be the highest since -(2-2)^2 = 0, exp(0) = 1
    assert almon_w[1] > almon_w[0]
    assert almon_w[1] > almon_w[2]
    # Check symmetry
    assert np.isclose(almon_w[0], almon_w[2])
    
    # Test application
    vals = np.array([100, 110, 120])
    weighted_sum = np.nansum(almon_w * vals)
    assert weighted_sum > 100 and weighted_sum < 120
