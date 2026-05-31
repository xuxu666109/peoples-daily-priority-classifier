# People’s Daily Page 1–3 Priority Classifier

This repository contains the final notebook for a time-aware Multi-BERT classifier that predicts whether a *People’s Daily* article belongs to Page 1–3 priority placement.

The project treats page placement as a proxy signal for editorial and political importance. The final model combines original Chinese text, political keyword features, time-aware context, focal-loss training, and a Multi-BERT soft ensemble.

## Project Motivation

*People’s Daily* is an important official channel through which policy priorities and political narratives are communicated. Instead of treating the task as ordinary news classification, this project aims to operationalize editorial priority as a measurable political signal.

The model predicts whether an article is likely to receive high-priority placement in Page 1–3.

## Key Design Choices

### 1. Page 1–3 Label Redesign

The initial setup only treated Page 1 articles as positive examples. Manual error analysis showed that many predicted false positives appeared on Pages 2–3 and still carried substantial political importance, such as:

- high-level diplomatic meetings
- central government policy implementation
- major official speeches
- leadership activities
- follow-up coverage of national policy themes

Therefore, the positive class was expanded from Page 1 to Pages 1–3. This reduced class imbalance and better reflected the newspaper’s editorial priority hierarchy.

### 2. Original Chinese Text

The model uses original Chinese text instead of translated English text. Official expressions such as “贯彻落实”, “中国式现代化”, and “中央政治局会议” carry institutional and political meanings that are better preserved in Chinese.

### 3. Political Signal Engineering

TF-IDF and distinction-score analysis were used as exploratory tools to discover words associated with high-priority articles. After manual review, candidate terms were organized into political keyword categories, including:

- top leadership
- central institutions
- official meetings
- policy concepts
- diplomacy
- military affairs

These curated keyword categories were converted into handcrafted features for the final classifier.

### 4. Time-Aware Context

The model includes the most recent previous front-page title within a 30-day window. This allows the classifier to consider ongoing policy narratives and recent editorial priorities.

### 5. Multi-BERT Soft Ensemble

The final system trains three Chinese BERT-family models:

- `hfl/chinese-roberta-wwm-ext`
- `google-bert/bert-base-chinese`
- `hfl/chinese-bert-wwm-ext`

Their prediction probabilities are combined through weighted soft ensemble to reduce single-model bias and improve prediction stability.

## Final Results

The final Multi-BERT soft ensemble achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.849 |
| Positive-class F1 | 0.767 |
| ROC-AUC | 0.910 |

The positive-class F1 improved from approximately 0.58 in earlier experiments to 0.767 in the final model.

## Repository Structure

```text
.
├── peoples_daily_priority_classifier_clean.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Data Requirements

The notebook expects the following files under `BASE_DIR`:

```text
train_df.parquet
val_df 1.parquet
test_df 1.parquet
```

Required columns:

| Column | Meaning |
|---|---|
| `scraped page number` | page number from the scraped article metadata |
| `title` | article title |
| `body` | article body |
| `published_at` | publication date |
| `labels` | generated label column used by the model |

The data files are not included in this repository.

## How to Run

### Option A: Run in Colab with Google Drive

Set the environment variable before running the setup cell:

```python
import os
os.environ["USE_GOOGLE_DRIVE"] = "1"
```

The notebook will mount Google Drive and use:

```text
/content/drive/MyDrive/Colab Notebooks
```

as the default `BASE_DIR`.

### Option B: Run Locally or from GitHub

Set `BASE_DIR` manually to the folder containing the `.parquet` files:

```python
import os
os.environ["BASE_DIR"] = "/path/to/your/data"
```

Then run the notebook cells in order.

## Notes

- Model checkpoints, raw data, downloaded Hugging Face models, and prediction CSV files should not be committed to GitHub.
- The notebook is designed for reproducibility and readability; outputs have been cleared for stable GitHub rendering.
