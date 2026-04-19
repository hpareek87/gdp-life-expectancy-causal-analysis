# Data Directory

## Structure

```
data/
├── raw/              # Raw API downloads (.gitignored — regenerate with build_dataset.py)
├── processed/        # Cleaned panel data (.gitignored — regenerate with build_dataset.py)
└── final/
    ├── master_dataset.csv       # 725 country-year obs × 133 variables (main analysis file)
    ├── data_dictionary.txt      # Variable descriptions and sources
    ├── data_quality_report.txt  # Missingness and validation summary
    └── reproducibility.json     # Pipeline run metadata
```

## Regenerating data

```bash
python -m src.data.build_dataset
```

This runs: API collection → cleaning → imputation → feature engineering → validation.
Takes ~20 minutes on first run (API rate limits). Subsequent runs use cached intermediate files.

## Master dataset

**File:** `data/final/master_dataset.csv`
**Shape:** 725 rows × 133 columns
**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `country` | str | Country name |
| `iso3` | str | ISO 3166-1 alpha-3 code |
| `year` | int | Year (2000–2024) |
| `income_group` | str | World Bank income classification |
| `life_expectancy` | float | Life expectancy at birth (years) |
| `gdp_per_capita_ppp` | float | GDP per capita, PPP 2017 intl $ |
| `health_exp_pct_gdp` | float | Health expenditure (% GDP) |
| `education_exp_pct_gdp` | float | Education expenditure (% GDP) |
| `fertility_rate` | float | Total fertility rate |
| `sanitation_access` | float | Basic sanitation access (%) |

See `data_dictionary.txt` for all 133 variables.
