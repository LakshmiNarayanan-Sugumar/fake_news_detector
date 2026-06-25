# Fake News Detector V1

## What is this?
A web app that detects whether a news article is real or fake.
This is V1 where I used traditional ML approaches — TF-IDF for feature extraction 
and Logistic Regression + Random Forest for classification.

## Why I built this?
Wanted to start with a classic ML approach before moving to transformers in V2.
Good way to understand the baseline before jumping into BERT.

## Results
**Logistic Regression**
- Accuracy: 98.99%
- F1 Score: 0.99

**Random Forest**
- Accuracy: 99.69%
- F1 Score: 1.00

- Dataset: ISOT Fake News Dataset
- Features: TF-IDF vectorization

## Tech Used
- Python
- Scikit-learn
- Streamlit
- TF-IDF

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## This project was upgraded to V2 with BERT
https://huggingface.co/spaces/LakshmiNarayanan-sugumar/fake-news-detector