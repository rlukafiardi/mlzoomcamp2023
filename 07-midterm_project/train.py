# Library
import pandas as pd
import numpy as np
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRFClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Parameters
eta = 0.01
eval_metric = 'auc'
max_depth = 20
min_child_weight = 1
n_estimators = 100
nthread = 20
objective = 'binary:logistic'
seed = 13
verbosity = 1
output_file = 'xgboost_model.bin'

# Data preparation
df = pd.read_csv("bank-full.csv", sep=";")
df.rename(columns={"y":"subscribed"}, inplace=True)

age_group = ['18-30', '31-45', '46-60', '>60']
age_bins = [18, 30, 45, 60, np.inf]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_group, include_lowest=True)

df["contacted"] = np.where(df["pdays"] >= 0, "yes", "no")
df["subscribed"] = (df["subscribed"] == "yes").astype(int)

numerical_cols = list(df.select_dtypes(include="number").columns)
numerical_cols.remove('subscribed')
categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=13)

# Training

# Create an instance of the OneHotEncoder with specified parameters
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit the OneHotEncoder to the categorical columns in the DataFrame df_model
ohe.fit(df[categorical_cols])

# Create a Repeated Stratified K-Fold cross-validator with 5 splits, 2 repeats, and a fixed random state
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=13)

# Initialize an empty list to store AUC scores for each fold
scores = []

# Extract labels ('y') and features ('X') from the DataFrame 'df_model'
y = df['subscribed'].values
X = df.drop(columns='subscribed')

# Iterate over folds using the Repeated Stratified K-Fold cross-validator
for fold_num, (train_index, test_index) in enumerate(rskf.split(X, y), 1):
    # Split the data into training and testing sets based on the current fold
    df_train, df_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # One-Hot Encode categorical features for both training and testing sets
    ohe_train = ohe.transform(df_train[categorical_cols])
    ohe_test = ohe.transform(df_test[categorical_cols])
    
    # Combine One-Hot Encoded features with numerical features
    X_train = np.column_stack([ohe_train, df_train[numerical_cols].values])
    X_test = np.column_stack([ohe_test, df_test[numerical_cols].values])
    
    # Fit the best model (assumed to be previously defined) on the training data
    model = XGBRFClassifier(eta=eta, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, objective='binary:logistic', eval_metric='auc', nthread=20, verbosity=1, seed=13)
    model.fit(X_train, y_train)
    
    # Generate predicted probabilities for the positive class on the test set
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # Calculate and round the AUC score for the current fold
    score = roc_auc_score(y_test, y_pred).round(4)
    
    # Append the AUC score to the 'scores' list
    scores.append(score)
    
    # Print the AUC score for the current fold
    print(f"AUC Score for fold {fold_num}: {score}")

# Calculate and round the average AUC score across all folds
average_score = np.mean(scores).round(4)

# Print the average AUC score across all folds
print(f"\nAverage AUC Score across all folds: {average_score}")

# Fit the best model (assumed to be previously defined) on the training data
model = XGBRFClassifier(eta=eta, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, objective='binary:logistic', eval_metric='auc', nthread=20, verbosity=1, seed=13)
model.fit(X_train, y_train)

# Generate predictions on the test set using the trained model
y_pred = model.predict(X_test)

# Generate predicted probabilities for the positive class on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate and round the AUC score using predicted probabilities
auc = round(roc_auc_score(y_test, y_pred_proba), 4)

# Print the AUC Score
print(f'AUC Score: {auc}')

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((ohe, model), f_out)

print(f'The model is saved to {output_file}')