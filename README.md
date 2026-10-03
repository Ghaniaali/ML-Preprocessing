Predicting Term-Deposit Subscription in Bank Telemarketing

CS-13410 Introduction to Machine Learning | Semester Project | Fall 2026 The University of Lahore, Department of Computer Science & IT

Team member	Roll No.
Ghania Ali	70149202
Syed Hassan Irtaza	70145652
Huda Bhukhari	70145623
Overview

A Portuguese bank sells term deposits through phone campaigns. Only about 12% of contacted clients subscribe, yet every call costs time and money. This project builds a machine learning model that estimates, before a call is made, how likely a client is to subscribe, so that calls can be prioritised towards likely subscribers.

The project follows one dataset through the full machine learning lifecycle, in five parts. Each part adds a clearly labelled section to a single, evolving Jupyter notebook.

Project Status
Part	Focus	Status
Part 1	Data understanding, preprocessing and leakage-free pipeline	Complete
Part 2	Naive Bayes vs decision tree baseline	Upcoming
Part 3	Ensembles, regularization and hyperparameter optimisation	Upcoming
Part 4	Neural network, unsupervised learning, interpretability and fairness	Upcoming
Part 5	Final integration, written report and viva	Upcoming
Dataset
Source: UCI Machine Learning Repository, Bank Marketing (file used: bank-full.csv)
Size: 45,211 clients, 16 input features and one target
Licence: CC BY 4.0, with no direct personal identifiers
Features:
Numeric: age, balance, day, campaign, pdays, previous
Categorical: job, marital, education, contact, month, poutcome
Binary: default, housing, loan
Target: y, whether the client subscribed to a term deposit (11.7% yes)
Problem Framing
Item	Definition
Task	Binary classification
Primary metrics	ROC-AUC and PR-AUC
Decision-level metric	F1 score on the positive ("yes") class
Why not accuracy	A model that always predicts "no" would be about 88% accurate while finding no subscribers
Part 1: Leakage-Free Preprocessing Pipeline

All preprocessing is built with scikit-learn Pipeline and ColumnTransformer, so no information from the test set can influence training.

Issue	Decision
Target leakage	duration (call length) is removed because it is only known after the call ends
Train/test split	Stratified 80/20 split performed before any learned preprocessing; test set held out
Missing contact, poutcome	Kept as their own "unknown" category, since the missingness is meaningful
Missing job, education	Most-frequent imputation
Missing numeric values	Median imputation
Outliers (balance, campaign, previous)	Clipped at the 1st and 99th percentile, learned from training data only
Sentinel value in pdays (-1)	previously_contacted flag created; pdays set to 0 where it was -1
Scaling	StandardScaler
Categorical encoding	One-hot encoding with handle_unknown="ignore"
Class imbalance	class_weight="balanced"
Feature selection	SelectKBest (ANOVA F-test, k = 20), refitted inside each cross-validation fold
Baseline Results

Stratified 5-fold cross-validation on the training set, using logistic regression as a placeholder estimator to validate the pipeline:

Metric	Mean	Std
ROC-AUC	0.749	0.001
PR-AUC	0.382	0.007
F1 (yes class)	0.356	0.005

A model with no skill would reach a PR-AUC of about 0.117 (the share of subscribers). The absolute scores are modest by design: the most predictive feature, duration, was removed to avoid leakage, so the model only uses information available before the call.

Exploratory Data Analysis

The notebook contains five visualisations, each with a written finding:

Target distribution (class imbalance)
Boxplots of skewed numeric features (outliers)
Subscription rate by job
Share of missing values per column
Correlation heatmap of numeric features

How to Run
Clone the repository:
bash
   git clone https://github.com/Ghaniaali/ML-Preprocessing.git
   cd ML-Preprocessing
Install the dependencies:
bash
   pip install -r requirements.txt
Make sure bank-full.csv is in the repository root or in a data/ folder. The notebook looks in both places.
Launch Jupyter and run the notebook from top to bottom:
bash
   jupyter notebook ML_Project_Part1.ipynb

Random seeds are fixed (random_state=42), so results are reproducible.

Limitations
The dataset is ordered roughly by time, but a stratified random split was used. A deployed model would predict future campaigns from past ones, so a time-based split would be a more realistic evaluation.
The baseline estimator is intentionally simple. Model comparison begins in Part 2.
References
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
UCI Machine Learning Repository: Bank Marketing dataset. https://archive.ics.uci.edu/dataset/222/bank+marketing
Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
Acknowledgements

The notebook structure and code were developed with assistance from Claude (Anthropic). The team ran, reviewed and validated all code and results.
