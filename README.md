# Bank Telemarketing Campaign Performance Analysis

## Purpose
Explore customer and campaign characteristics associated with subscription outcomes and demonstrate an imbalanced classification workflow.

## Data notice
`data/telemarketing_demo.csv` is **synthetic demonstration data** generated for this portfolio. It is not a bank's customer data and does not reproduce a real campaign. Treat all results as educational examples.

## Questions
- What is the demo subscription rate?
- How does subscription vary by contact method and job category?
- How is the number of campaign contacts associated with subscription?
- How does a class-weighted classifier perform on a held-out sample?

## Method
The script uses a stratified train/test split, a validation split to select a probability threshold, class-weighted logistic regression and ROC AUC, average precision, precision, recall and F1 reporting. Call duration is excluded because it is only known after a call and would leak post-call information into a pre-call targeting model.

## Tools
Python, pandas, Matplotlib and scikit-learn.

## Run
```bash
pip install -r requirements.txt
python src/analyze_campaign.py
```
Outputs are written to `outputs/`.

## Limitations
The synthetic data-generating process is artificial. Metrics and patterns are not evidence about real customers or a real bank. Threshold choice should reflect campaign capacity and the relative costs of missed subscribers and unnecessary contacts.
