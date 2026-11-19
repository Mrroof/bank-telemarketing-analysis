# Bank Telemarketing Campaign Performance Analysis

## Purpose
Explore customer and campaign characteristics associated with subscription outcomes.

## Data notice
`data/telemarketing_demo.csv` is **synthetic demonstration data** generated for this portfolio. It is not a bank's customer data and does not reproduce a real campaign. Treat all results as educational examples.

## Questions
- What is the demo subscription rate?
- How does subscription vary by contact method and job category?
- How is the number of campaign contacts associated with subscription?
- How does a basic classification model perform on a held-out sample?

## Tools
Python, pandas, NumPy, Matplotlib and scikit-learn.

## Run
```bash
pip install -r requirements.txt
python src/analyze_campaign.py
```
Outputs are written to `outputs/`.

## Limitations
The synthetic data-generating process is artificial. Metrics and patterns are not evidence about real customers or a real bank.
