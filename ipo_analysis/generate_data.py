
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 400

# Generate synthetic IPO data
data = {
    'company_age': np.random.exponential(8, n_samples),
    'employees': np.random.lognormal(6, 1.5, n_samples),
    'revenue_millions': np.random.lognormal(4, 1.8, n_samples),
    'revenue_growth_rate': np.random.normal(30, 25, n_samples),
    'ebitda_margin': np.random.normal(-5, 15, n_samples),
    'industry_sector': np.random.choice(['Tech', 'Healthcare', 'Finance', 'Consumer', 'Industrial', 'Energy'], n_samples),
    'offer_price': np.random.lognormal(2.5, 0.6, n_samples),
    'shares_offered_millions': np.random.lognormal(2, 1, n_samples),
    'underwriter_rank': np.random.randint(1, 11, n_samples),
    'venture_backed': np.random.binomial(1, 0.6, n_samples),
    'lockup_period_days': np.random.choice([90, 180, 270, 365], n_samples),
    'market_volatility_index': np.random.normal(18, 6, n_samples),
    'sector_performance_30d': np.random.normal(2, 8, n_samples),
    'ipo_month': np.random.randint(1, 13, n_samples),
}

df = pd.DataFrame(data)

# Create target with some signal
score = (
    df['revenue_growth_rate'] * 0.02 +
    df['underwriter_rank'] * 0.15 +
    df['venture_backed'] * 0.3 -
    df['market_volatility_index'] * 0.05 +
    df['sector_performance_30d'] * 0.08 +
    np.random.normal(0, 1, n_samples)
)

df['q1_return'] = (score > score.median()).astype(int)

# Add missing values
missing_mask = np.random.random(df.shape) < 0.1
df = df.mask(missing_mask)

df.to_csv('ipo_data.csv', index=False)
