# Predicting Term-Deposit Subscription in Bank Telemarketing

**CS-13410 Introduction to Machine Learning | Semester Project | Fall 2026**
The University of Lahore, Department of Computer Science & IT

| Team member | Roll No. |
|---|---|
| Ghania Ali | 70149202 |
| Syed Hassan Irtaza | 70145652 |
| Huda Bhukhari | 70145623 |

---

## Overview

A Portuguese bank sells term deposits through phone campaigns. Only about 12% of contacted clients subscribe, yet every call costs time and money. This project builds a machine learning model that estimates, **before a call is made**, how likely a client is to subscribe, so that calls can be prioritised towards likely subscribers.

The project follows one dataset through the full machine learning lifecycle, in five parts. Each part adds a clearly labelled section to a single, evolving Jupyter notebook.

## Project Status

| Part | Focus | Status |
|---|---|---|
| **Part 1** | Data understanding, preprocessing and leakage-free pipeline | Complete |
| **Part 2** | Naive Bayes vs decision tree baseline | Complete |
| Part 3 | Ensembles, regularization and hyperparameter optimisation | Upcoming |
| Part 4 | Neural network, unsupervised learning, interpretability and fairness | Upcoming |
| Part 5 | Final integration, written report and viva | Upcoming |

## Dataset

- **Source:** [UCI Machine Learning Repository, Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (file used: `bank-full.csv`)
- **Size:** 45,211 clients, 16 input features and one target
- **Licence:** CC BY 4.0, with no direct personal identifiers
- **Features:**
  - Numeric: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
  - Categorical: `job`, `marital`, `education`, `contact`, `month`, `poutcome`
  - Binary: `default`, `housing`, `loan`
- **Target:** `y`, whether the client subscribed to a term deposit (11.7% yes)

## Problem Framing

| Item | Definition |
|---|---|
| Task | Binary classification |
| Primary metrics | ROC-AUC and PR-AUC |
| Decision-level metric | F1 score on the positive ("yes") class |
| Why not accuracy | A model that always predicts "no" would be about 88% accurate while finding no subscribers |

## Part 1: Leakage-Free Preprocessing Pipeline

All preprocessing is built with scikit-learn `Pipeline` and `ColumnTransformer`, so no information from the test set can influence training.

| Issue | Decision |
|---|---|
| Target leakage | `duration` (call length) is removed because it is only known after the call ends |
| Train/test split | Stratified 80/20 split performed **before** any learned preprocessing; test set held out |
| Missing `contact`, `poutcome` | Kept as their own `"unknown"` category, since the missingness is meaningful |
| Missing `job`, `education` | Most-frequent imputation |
| Missing numeric values | Median imputation |
| Outliers (`balance`, `campaign`, `previous`) | Clipped at the 1st and 99th percentile, learned from training data only |
| Sentinel value in `pdays` (-1) | `previously_contacted` flag created; `pdays` set to 0 where it was -1 |
| Scaling | `StandardScaler` |
| Categorical encoding | One-hot encoding with `handle_unknown="ignore"` |
| Class imbalance | `class_weight="balanced"` |
| Feature selection | `SelectKBest` (ANOVA F-test, k = 20), refitted inside each cross-validation fold |

### Baseline Results

Stratified 5-fold cross-validation on the training set, using logistic regression as a placeholder estimator to validate the pipeline:

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.749 | 0.001 |
| PR-AUC | 0.382 | 0.007 |
| F1 (yes class) | 0.356 | 0.005 |

A model with no skill would reach a PR-AUC of about 0.117 (the share of subscribers). The absolute scores are modest by design: the most predictive feature, `duration`, was removed to avoid leakage, so the model only uses information available before the call.

### Exploratory Data Analysis

The notebook contains five visualisations, each with a written finding:

1. Target distribution (class imbalance)
2. Boxplots of skewed numeric features (outliers)
3. Subscription rate by job
4. Share of missing values per column
5. Correlation heatmap of numeric features

## Part 2: Naive Bayes vs Decision Tree Baseline

Part 2 trains and compares probabilistic and tree-based models on the Part 1 data, using the same stratified 5-fold cross-validation folds for every model. The test set remains untouched.

**Models:** Gaussian Naive Bayes, Categorical Naive Bayes (discretised features; natural and uniform class priors), an unrestricted decision tree, a depth-limited decision tree, and a decision tree whose depth is tuned with nested cross-validation, all compared with a majority-class baseline.

**Hand-worked split:** information gain (0.1621 bits) and Gini gain (0.1013) for the split `poutcome == "success"` on a 20-client sample, verified in code and then checked against the root split chosen by scikit-learn on the full training set.

### Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.883 | 0.000 | 0.000 | 0.000 | 0.500 | 0.117 |
| Gaussian NB | 0.828 | 0.330 | 0.458 | 0.383 | 0.740 | 0.334 |
| Categorical NB | 0.871 | 0.435 | 0.355 | 0.391 | 0.752 | 0.375 |
| Categorical NB (uniform priors) | 0.725 | 0.243 | 0.637 | 0.352 | 0.752 | 0.375 |
| Decision tree (unlimited depth) | 0.837 | 0.306 | 0.310 | 0.308 | 0.608 | 0.176 |
| Decision tree (max_depth=8) | 0.810 | 0.328 | 0.572 | 0.415 | 0.754 | 0.372 |
| Decision tree (depth tuned, nested CV) | 0.815 | 0.337 | 0.561 | 0.418 | 0.752 | 0.362 |

### Key Findings

- Accuracy is misleading: the majority-class baseline has the highest accuracy (88.3%) while finding no subscribers.
- Categorical NB and the depth-tuned decision tree are effectively tied on ranking quality (ROC-AUC about 0.75, PR-AUC about 0.37); the difference is smaller than the fold-to-fold variation.
- An unrestricted tree overfits badly (training ROC-AUC 1.0, validation about 0.61). Validation performance peaks at a depth of about 8.
- Categorical NB outperforms Gaussian NB because discretisation suits skewed and categorical features.
- The outcome of the previous campaign (`poutcome`) is the strongest signal and is the root split of the tree.
- The tree reaches higher recall and F1 largely through class weighting (a threshold effect); Naive Bayes is faster and more stable, and the tree is easier to explain as rules.

```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ghaniaali/ML-Preprocessing.git
   cd ML-Preprocessing
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the dataset is at `bank/bank-full.csv` (included in the repository). The notebook loads it from there.
4. Launch Jupyter and run the notebook from top to bottom:
   ```bash
   jupyter notebook ML_Project.ipynb
   ```


## Limitations

- The dataset is ordered roughly by time, but a stratified random split was used. A deployed model would predict future campaigns from past ones, so a time-based split would be a more realistic evaluation.
- Models in Part 2 use default or lightly tuned settings; thresholds are not tuned and imbalance is handled only through class weights and priors. Part 3 addresses ensembles, regularization and systematic hyperparameter optimisation.

## References

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22-31.
- UCI Machine Learning Repository: Bank Marketing dataset. https://archive.ics.uci.edu/dataset/222/bank+marketing
- Pedregosa et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.

## Acknowledgements

The notebook structure and code were developed with assistance from Claude (Anthropic). The team ran, reviewed and validated all code and results.
