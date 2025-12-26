# %%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# import seaborn as sns

# Load data
df = pd.read_json('data/web_search_std_rag_traindata_weibo_feature_eng.json', lines=True)

# View basic data information
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData description:")
print(df.describe())
print("\nData types:")
print(df.dtypes)
print("\nMissing values count:")
print(df.isnull().sum())


# %%
# Display original data shape
print(f"Original data shape: {df.shape}")

# Check data type of ExternalConsistencyAnalysis column
print("Original data type of ExternalConsistencyAnalysis column:", df['ExternalConsistencyAnalysis'].dtype)

# Create a mask to mark rows that can be converted to float
valid_float_mask = []
for value in df['ExternalConsistencyAnalysis']:
    # NaN values are valid floats
    if pd.isna(value):
        valid_float_mask.append(True)
        continue
    
    # Try to convert to float
    try:
        float(value)
        valid_float_mask.append(True)
    except (ValueError, TypeError):
        valid_float_mask.append(False)

# Apply mask, keep only rows that can be converted to float
df_clean = df[valid_float_mask].copy()

# Convert ExternalConsistencyAnalysis column to float type
df_clean['ExternalConsistencyAnalysis'] = df_clean['ExternalConsistencyAnalysis'].astype(float)

# Display cleaned data shape and type
print(f"Cleaned data shape: {df_clean.shape}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} rows of non-float64 data")
print("Data type of ExternalConsistencyAnalysis column after cleaning:", df_clean['ExternalConsistencyAnalysis'].dtype)


# %%
import json

# Load data
df_test = pd.read_json('data/web_search_std_rag_testdata_weibocovide_feature_eng.json', lines=True)

# Read LLM prediction results
with open('data/web_search_rag_predict_glm4_all.json', 'r') as f:
    lines = f.readlines()
# Convert each line to JSON format
data = [json.loads(line) for line in lines]

# Convert "yes" and "no" in predict to 1 and 0
preds = [1 if "yes" in str(d['predict']).lower() else 0 for d in data]
df_test['llm_predict'] = preds

# View basic data information
print("数据形状:", df_test.shape)
print("\nFirst 5 rows:")
print(df_test.head())
print("\nData description:")
print(df_test.describe())
print("\nData types:")
print(df_test.dtypes)
print("\nMissing values count:")
print(df_test.isnull().sum())

# %%
df = df_clean

# Define feature list
features = [
    "CoerciveLanguageAnalysis",
    "DivisiveContentIdentification",
    "ManipulativeRhetoricAnalysis",
    "AbsolutistLanguageDetection",
    "FactualConsistencyVerification",
    "LogicalFallacyIdentification",
    "AttributionAndSourceEvaluation",
    "ConspiracyTheoryNa",
    "EmotionalAppealAnalysis",
    "PseudoscientificLanguageIdentification",
    "CallActionAssessment",
    "AuthorityImpersonationDetection",
    "BotActivitySignDetection",
    "UserReactionAssessment",
    "DisseminationModificationTracking",
    "SourceCredibilityAssessment",
    "FactualAccuracyVerification",
    "InformationCompletenessCheck",
    "ExternalConsistencyAnalysis",
    "ExpertConsensusAlignment"
]

# Hypothesis testing - including variance homogeneity test and appropriate t-test
print("\nHypothesis testing results (including variance homogeneity test):")
print("-" * 100)
test_results = []

for feature in features:
    # Separate two groups of data
    group0 = df[df['label'] == 0][feature]
    group1 = df[df['label'] == 1][feature]
    
    # Perform Levene test (variance homogeneity test)
    levene_stat, levene_p = stats.levene(group0, group1)
    
    # Select appropriate t-test based on variance homogeneity test results
    if levene_p > 0.05:  # Equal variance
        t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=True)
        t_test_type = "Student's t-test (equal variance)"
    else:  # Unequal variance
        t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False)
        t_test_type = "Welch's t-test (unequal variance)"
    
    # Store results
    test_results.append({
        'Feature': feature,
        'Levene_statistic': levene_stat,
        'Levene_p_value': levene_p,
        'Equal_variance': levene_p > 0.05,
        'T_test_type': t_test_type,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05  # Significance level set to 0.05
    })

# Convert to DataFrame and display
test_results_df = pd.DataFrame(test_results)
print(test_results_df)

# %%
# Visualize p-values
plt.figure(figsize=(12, 8))
pvalues = test_results_df.set_index('Feature')['p_value'].sort_values()
ax = pvalues.plot(kind='barh')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xscale('log')  # Use logarithmic scale

plt.title('P-values of Hypothesis Tests for Each Feature')
plt.xlabel('P-value (log scale)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
# Calculate effect size (Cohen's d)
print("\nEffect size (Cohen's d) results:")
print("-" * 80)
effect_sizes = []

for feature in features:
    # Separate two groups of data
    group0 = df[df['label'] == 0][feature]
    group1 = df[df['label'] == 1][feature]
    
    # Calculate mean and standard deviation
    mean0, mean1 = group0.mean(), group1.mean()
    std0, std1 = group0.std(), group1.std()
    n0, n1 = len(group0), len(group1)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n0-1)*std0**2 + (n1-1)*std1**2) / (n0 + n1 - 2))
    
    # Calculate Cohen's d
    cohens_d = (mean1 - mean0) / pooled_std
    
    # Store results
    effect_sizes.append({
        'Feature': feature,
        'Cohens_d': cohens_d,
        'Effect_size': 'small' if abs(cohens_d) < 0.2 else ('medium' if abs(cohens_d) < 0.5 else 'large')
    })

# Convert to DataFrame and display
effect_sizes_df = pd.DataFrame(effect_sizes)
print(effect_sizes_df)

# %%
# Visualize effect sizes
plt.figure(figsize=(12, 8))
effect_sizes_df.set_index('Feature')['Cohens_d'].sort_values().plot(kind='barh')
plt.axvline(x=0, color='k', linestyle='-')
plt.axvline(x=0.2, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=-0.2, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=-0.5, color='r', linestyle='--', alpha=0.5)
plt.title('Effect Size (Cohen\'s d) for Each Feature')
plt.xlabel('Cohen\'s d')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
# Define feature list
features = [
    "CoerciveLanguageAnalysis",
    "DivisiveContentIdentification",
    "ManipulativeRhetoricAnalysis",
    "AbsolutistLanguageDetection",
    "FactualConsistencyVerification",
    "LogicalFallacyIdentification",
    "AttributionAndSourceEvaluation",
    "ConspiracyTheoryNa",
    "EmotionalAppealAnalysis",
    "PseudoscientificLanguageIdentification",
    "CallActionAssessment",
    "AuthorityImpersonationDetection",
    "BotActivitySignDetection",
    "UserReactionAssessment",
    "DisseminationModificationTracking",
    "SourceCredibilityAssessment",
    "FactualAccuracyVerification",
    "InformationCompletenessCheck",
    "ExternalConsistencyAnalysis",
    "ExpertConsensusAlignment"
]

# Prepare features and target variables
X_train = df_clean[features]
y_train = df_clean['label']

X_test = df_test[features]
y_test = df_test['label']

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Basic logistic regression model
print("\nBuilding basic logistic regression model...")
base_lr = LogisticRegression(random_state=42, max_iter=1000)
base_lr.fit(X_train, y_train)

# Basic decision tree model
print("\nBuilding basic decision tree model...")
base_dt = DecisionTreeClassifier(random_state=42)
base_dt.fit(X_train, y_train)

# Basic support vector machine model
print("\nBuilding basic support vector machine model...")
base_svm = SVC()
base_svm.fit(X_train, y_train)

# %%
# Evaluate basic models
y_pred_lr = base_lr.predict(X_test)
df_test['lr_predict'] = y_pred_lr

y_pred_dt = base_dt.predict(X_test)
df_test['dt_predict'] = y_pred_dt

y_pred_svm = base_svm.predict(X_test)
df_test['svm_predict'] = y_pred_svm

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\nBasic logistic regression model evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")

# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in y_pred_lr]
neg_labels = [1 if l == 0 else 0 for l in y_test]
neg_f1 = f1_score(neg_labels, neg_preds)
print(f"Negative prediction F1 score: {neg_f1:.4f}")

print(f"F1 score: {f1_score(y_test, y_pred_lr):.4f}")

# %%
print("\nBasic decision tree model evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in y_pred_dt]
neg_labels = [1 if l == 0 else 0 for l in y_test]
neg_f1 = f1_score(neg_labels, neg_preds)
print(f"Negative prediction F1 score: {neg_f1:.4f}")
print(f"F1 score: {f1_score(y_test, y_pred_dt):.4f}")


# %%
print("\nBasic support vector machine model evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.4f}")
# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in y_pred_svm]
neg_labels = [1 if l == 0 else 0 for l in y_test]
neg_f1 = f1_score(neg_labels, neg_preds)
print(f"Negative prediction F1 score: {neg_f1:.4f}")
print(f"F1 score: {f1_score(y_test, y_pred_svm):.4f}")

# %%
y_pred_llm = df_test['llm_predict']

print("\nLLM prediction evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_llm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_llm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_llm):.4f}")
# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in y_pred_llm]
neg_labels = [1 if l == 0 else 0 for l in y_test]
neg_f1 = f1_score(neg_labels, neg_preds)
print(f"Negative prediction F1 score: {neg_f1:.4f}")
print(f"F1 score: {f1_score(y_test, y_pred_llm):.4f}")

# %%
# Ensemble the above models using voting method with pandas

# Create a new column to store voting results
df_test['ensemble_predict'] = df_test.apply(lambda row: 1 if row["llm_predict"] + row["dt_predict"] + row["svm_predict"] > 1 else 0, axis=1)

# %%
ensemble_predict = df_test['ensemble_predict']
print("\nEnsemble model evaluation:")
print(f"Accuracy: {accuracy_score(y_test, ensemble_predict):.4f}")
print(f"Precision: {precision_score(y_test, ensemble_predict):.4f}")
print(f"Recall: {recall_score(y_test, ensemble_predict):.4f}")
# Calculate F1 score for negative predictions
neg_preds = [1 if p == 0 else 0 for p in ensemble_predict]
neg_labels = [1 if l == 0 else 0 for l in y_test]
neg_f1 = f1_score(neg_labels, neg_preds)
print(f"Negative prediction F1 score: {neg_f1:.4f}")
print(f"F1 score: {f1_score(y_test, ensemble_predict):.4f}")

# %%



