[![Portfolio validation](https://github.com/Mrroof/bank-telemarketing-analysis/actions/workflows/portfolio-validation.yml/badge.svg)](https://github.com/Mrroof/bank-telemarketing-analysis/actions)

# Bank Telemarketing Campaign Analysis

> **Portfolio case study · Python · pandas · scikit-learn · Matplotlib**

## Executive summary
This project explores patterns in a synthetic term-deposit campaign dataset and demonstrates how to evaluate a classifier when positive outcomes are relatively uncommon. It emphasizes recall, precision and average precision rather than relying on accuracy alone.

## Business questions
- How does the demo subscription rate vary by contact channel and job category?
- How does campaign contact frequency relate to subscription?
- How well can a pre-call model identify likely subscribers?

## Demonstration findings
- Overall synthetic subscription rate: **14.9%** (1,200 records).
- Logistic regression achieved validation average precision of **0.425**, compared with **0.418** for the random-forest candidate; logistic regression was selected.
- On the held-out test set, the selected model achieved **0.635 ROC AUC**, **0.316 average precision** and **77% accuracy**.
- At the validation-selected threshold of 0.65, subscriber precision was **30%** and recall was **38%** (17 true positives from 45 actual subscribers).

![Synthetic telemarketing model performance metrics](docs/telemarketing-performance.svg)

The benchmark confirms that the more complex random forest did not improve validation ranking performance on this sample. The selected classifier remains an exploratory ranking aid, not a high-performing or deployment-ready targeting model.

## Method
1. Explore subscription rates by contact type and job category.
2. Split the records into training, validation and untouched test samples using stratification.
3. Compare logistic regression and random forest using validation average precision.
4. Select a probability threshold using validation F1.
5. Refit the selected model on all training data.
6. Evaluate ROC AUC, average precision, precision, recall, F1 and the confusion matrix on the held-out test set.

**Leakage control:** call duration is excluded because it is only known after a call and would leak post-call information into a pre-call targeting model.

## Run locally
```bash
pip install -r requirements.txt
python src/analyze_campaign.py
```

Outputs are written to `outputs/`, including the model comparison, classification report and precision-recall curve.

## Limitations and responsible use
The CSV is synthetic and does not represent a real bank campaign or real customers. The performance metrics are illustrative. A practical campaign model needs representative data, time-aware validation, calibration, a threshold based on campaign capacity and contact costs, and monitoring for drift and unfair impact.

## Repository structure

```text
data/telemarketing_demo.csv          # Synthetic campaign records
src/analyze_campaign.py              # Analysis and model comparison
docs/telemarketing-performance.svg   # Report-ready model summary
requirements.txt                     # Python dependencies
.github/workflows/                   # Automated repository validation
```

## Reproducibility

Run the command above from the repository root. The script regenerates the analysis outputs locally in `outputs/`; generated outputs are not committed by default. The GitHub Actions validation workflow checks the project on repository updates.

## Business insights

- The demo subscription rate is **14.9%**, so a model that simply predicts "no" would appear accurate while failing to identify potential subscribers.
- Logistic regression slightly outperformed the random-forest candidate on validation average precision (**0.425 vs 0.418**). The more complex model did not earn its added complexity on this sample.
- On the untouched test set, the selected model recorded **0.316 average precision** and **0.635 ROC AUC**. At the validation-selected threshold of **0.65**, subscriber precision was **30%** and recall was **38%**.
- This operating point misses many actual subscribers and sends a substantial share of contacted leads that do not subscribe. It is not strong enough to justify automated campaign allocation.

## Practical recommendations

1. Treat the model as an exploratory prioritisation aid, not an automated decision-maker.
2. Before any pilot, choose a threshold using campaign capacity, contact cost and the relative value of a successful subscription.
3. Track lift and conversion by score band in a controlled pilot, and compare against the existing campaign selection process.
4. Re-test on a larger, time-ordered dataset and monitor performance across customer groups.
5. Keep post-call fields such as call duration out of any pre-call scoring process.
