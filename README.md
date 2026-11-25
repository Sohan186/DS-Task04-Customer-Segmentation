# Customer Segmentation Analysis (K-Means Clustering)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green)

**IncodeVision Data Science Internship | Task 04 (Track Code: DS)**

Groups mall customers into segments using clustering so a business can target each group differently.

**Dataset:** Mall Customers (200 customers: Gender, Age, Annual Income in k$, Spending Score 1-100). No missing values or duplicates.

## Approach

1. **EDA:** distributions of Age, Income, Spending Score and a correlation heatmap.
2. **Scaling:** `StandardScaler` on Age, Income and Spending Score so no feature dominates.
3. **Choosing k:** elbow method plus silhouette score for k = 2 to 10. Silhouette is highest at **k = 6** (0.428).
4. **Model:** K-Means (`k=6`, `n_init=10`, `random_state=42`).
5. **Visualisation:** Income vs Spending scatter and a 2D PCA view.
6. **Profiling:** average age, income, spending and gender mix for each segment.

## Customer segments

| Cluster | Customers | Avg age | Avg income (k$) | Avg spending | Who they are |
|---|---|---|---|---|---|
| 3 | 39 | 32.7 | 86.5 | 82.1 | **High income, high spending.** Best customers: loyalty and premium offers |
| 4 | 23 | 25.0 | 25.3 | 77.6 | **Low income, high spending, young.** Trend-driven: discounts, new arrivals |
| 2 | 33 | 41.9 | 88.9 | 17.0 | **High income, low spending.** Big opportunity: personalised campaigns |
| 5 | 21 | 45.5 | 26.3 | 19.4 | **Low income, low spending.** Low priority, keep costs low |
| 0 | 45 | 56.3 | 54.3 | 49.1 | **Average income and spending, older.** Steady base: regular offers |
| 1 | 39 | 26.8 | 57.1 | 48.1 | **Average income and spending, younger.** Growth group: engage early |

## Project structure

```
DS-Task04-Customer-Segmentation/
├── customer_segmentation.ipynb   # notebook with outputs (start here)
├── customer_segmentation.py
├── requirements.txt
├── data/Mall_Customers.csv
└── outputs/
    ├── 01_distributions.png
    ├── 02_correlation.png
    ├── 03_elbow_silhouette.png
    ├── 04_clusters_income_spending.png
    ├── 05_clusters_pca.png
    ├── 06_segment_spending.png
    ├── customers_with_segments.csv
    ├── segment_profile.csv
    └── k_selection.csv
```

## How to run

Open the notebook for the full walkthrough with outputs: [`customer_segmentation.ipynb`](customer_segmentation.ipynb). Or run the script:

```bash
pip install -r requirements.txt
python customer_segmentation.py
```

## Visuals

![Elbow and silhouette](outputs/03_elbow_silhouette.png)
![Segments](outputs/04_clusters_income_spending.png)
![PCA](outputs/05_clusters_pca.png)

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn.
