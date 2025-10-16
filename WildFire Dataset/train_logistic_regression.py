import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import time
import pickle

# Input paths
TRAINING_PATH = "features_array_training_top81_cov85.csv"
VALIDATION_PATH = "features_array_validation_top81.csv"

# Output paths for models and results
RESULTS_PATH = "models/LR/logistic_regression_results.txt"
MODEL_L1_PATH = "models/LR/model_l1_lasso.pkl"
MODEL_L2_PATH = "models/LR/model_l2_ridge.pkl"
MODEL_ELASTIC_PATH = "models/LR/model_elasticnet.pkl"
SCALER_PATH = "models/LR/scaler.pkl"
BEST_MODEL_PATH = "models/LR/best_model.pkl"

print("="*80)
print("LOGISTIC REGRESSION TRAINING WITH REGULARIZATION")
print("="*80)
print("\nWhat is Regularization?")
print("-" * 80)
print("Regularization prevents overfitting by penalizing overly complex models.")
print("It's like telling the model: 'Don't try too hard to fit every tiny detail")
print("in the training data - focus on the general patterns that will work on new data.'")
print()
print("Two main types:")
print("  L1 (Lasso): Forces unimportant feature coefficients to ZERO")
print("              → Automatic feature selection + prevents overfitting")
print("  L2 (Ridge): Shrinks all coefficients toward zero (but not to zero)")
print("              → Handles correlated features well + prevents overfitting")
print("="*80)

start_time = time.time()

# Step 1: Load data
print("\n" + "="*80)
print("STEP 1: LOADING TRAINING AND VALIDATION DATA")
print("="*80)

print(f"\nLoading training set from: {TRAINING_PATH}")
df_train = pd.read_csv(TRAINING_PATH, sep="\t", low_memory=False)
print(f"  ✓ Loaded: {df_train.shape[0]:,} rows × {df_train.shape[1]} columns")

print(f"\nLoading validation set from: {VALIDATION_PATH}")
df_val = pd.read_csv(VALIDATION_PATH, sep="\t", low_memory=False)
print(f"  ✓ Loaded: {df_val.shape[0]:,} rows × {df_val.shape[1]} columns")

# Separate features and target
X_train = df_train.drop(columns=['fire_spread'])
y_train = df_train['fire_spread']

X_val = df_val.drop(columns=['fire_spread'])
y_val = df_val['fire_spread']

print(f"\nTraining set:")
print(f"  Features: {X_train.shape[1]}")
print(f"  Samples:  {X_train.shape[0]:,}")
print(f"  Distribution: {(y_train == 0).sum():,} no-spread, {(y_train == 1).sum():,} spread")

print(f"\nValidation set:")
print(f"  Features: {X_val.shape[1]}")
print(f"  Samples:  {X_val.shape[0]:,}")
print(f"  Distribution: {(y_val == 0).sum():,} no-spread, {(y_val == 1).sum():,} spread")

# Step 2: Standardize features (CRITICAL for regularization!)
print("\n" + "="*80)
print("STEP 2: STANDARDIZING FEATURES")
print("="*80)
print("\nWhy standardize?")
print("  Regularization is sensitive to feature scales. If one feature has values")
print("  0-1000 and another has 0-1, regularization will unfairly penalize the large one.")
print("  Standardization makes all features have mean=0 and std=1, so they're comparable.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"\n✓ Features standardized")
print(f"  Training set scaled: {X_train_scaled.shape}")
print(f"  Validation set scaled: {X_val_scaled.shape}")

# Save the scaler (needed for future predictions!)
print(f"\nSaving scaler to: {SCALER_PATH}")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Scaler saved (you'll need this to standardize new data!)")

# Step 3: Train models with different regularization methods
print("\n" + "="*80)
print("STEP 3: TRAINING MODELS WITH CROSS-VALIDATION")
print("="*80)

results = []

# Model 1: L1 (Lasso) Regularization - Best for feature selection
print("\n" + "-"*80)
print("MODEL 1: L1 (LASSO) REGULARIZATION")
print("-"*80)
print("What it does:")
print("  - Forces some feature coefficients to exactly ZERO")
print("  - Automatically selects the most important features")
print("  - Good when you suspect some of your 81 features are irrelevant")
print()
print("Training with 5-fold cross-validation to find best regularization strength...")

l1_start = time.time()

# LogisticRegressionCV automatically finds best C (inverse regularization strength)
# Smaller C = stronger regularization
# Cs=10 means it will try 10 different values
model_l1 = LogisticRegressionCV(
    Cs=10,                    # Try 10 different regularization strengths
    cv=5,                     # 5-fold cross-validation
    penalty='l1',             # L1 (Lasso) regularization
    solver='liblinear',       # Required for L1
    scoring='f1',             # Optimize for F1 score (good for imbalanced data)
    max_iter=1000,
    random_state=42,
    n_jobs=-1                 # Use all CPU cores
)

model_l1.fit(X_train_scaled, y_train)
l1_time = time.time() - l1_start

# Get predictions
y_train_pred_l1 = model_l1.predict(X_train_scaled)
y_val_pred_l1 = model_l1.predict(X_val_scaled)
y_val_proba_l1 = model_l1.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
train_acc_l1 = accuracy_score(y_train, y_train_pred_l1)
val_acc_l1 = accuracy_score(y_val, y_val_pred_l1)
val_precision_l1 = precision_score(y_val, y_val_pred_l1)
val_recall_l1 = recall_score(y_val, y_val_pred_l1)
val_f1_l1 = f1_score(y_val, y_val_pred_l1)
val_auc_l1 = roc_auc_score(y_val, y_val_proba_l1)

# Count non-zero coefficients (selected features)
non_zero_coefs_l1 = np.sum(model_l1.coef_ != 0)

print(f"\n✓ Training complete in {l1_time:.2f} seconds")
print(f"\nBest regularization strength (C): {model_l1.C_[0]:.4f}")
print(f"Features selected (non-zero coefficients): {non_zero_coefs_l1}/{X_train.shape[1]}")
print(f"\nTraining Performance:")
print(f"  Accuracy: {train_acc_l1:.4f}")
print(f"\nValidation Performance:")
print(f"  Accuracy:  {val_acc_l1:.4f}")
print(f"  Precision: {val_precision_l1:.4f}")
print(f"  Recall:    {val_recall_l1:.4f}")
print(f"  F1 Score:  {val_f1_l1:.4f}")
print(f"  AUC-ROC:   {val_auc_l1:.4f}")

results.append({
    'Model': 'L1 (Lasso)',
    'C': model_l1.C_[0],
    'Features_Selected': non_zero_coefs_l1,
    'Train_Accuracy': train_acc_l1,
    'Val_Accuracy': val_acc_l1,
    'Val_Precision': val_precision_l1,
    'Val_Recall': val_recall_l1,
    'Val_F1': val_f1_l1,
    'Val_AUC': val_auc_l1,
    'Training_Time': l1_time
})

# Save L1 model
print(f"\nSaving L1 model to: {MODEL_L1_PATH}")
with open(MODEL_L1_PATH, 'wb') as f:
    pickle.dump(model_l1, f)
print(f"  ✓ L1 model saved")

# Model 2: L2 (Ridge) Regularization - Best for correlated features
print("\n" + "-"*80)
print("MODEL 2: L2 (RIDGE) REGULARIZATION")
print("-"*80)
print("What it does:")
print("  - Shrinks all coefficients but keeps them non-zero")
print("  - Handles correlated features better than L1")
print("  - Good when many of your 81 features are somewhat important")
print()
print("Training with 5-fold cross-validation to find best regularization strength...")

l2_start = time.time()

model_l2 = LogisticRegressionCV(
    Cs=10,
    cv=5,
    penalty='l2',             # L2 (Ridge) regularization
    solver='lbfgs',           # Good general-purpose solver
    scoring='f1',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

model_l2.fit(X_train_scaled, y_train)
l2_time = time.time() - l2_start

# Get predictions
y_train_pred_l2 = model_l2.predict(X_train_scaled)
y_val_pred_l2 = model_l2.predict(X_val_scaled)
y_val_proba_l2 = model_l2.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
train_acc_l2 = accuracy_score(y_train, y_train_pred_l2)
val_acc_l2 = accuracy_score(y_val, y_val_pred_l2)
val_precision_l2 = precision_score(y_val, y_val_pred_l2)
val_recall_l2 = recall_score(y_val, y_val_pred_l2)
val_f1_l2 = f1_score(y_val, y_val_pred_l2)
val_auc_l2 = roc_auc_score(y_val, y_val_proba_l2)

print(f"\n✓ Training complete in {l2_time:.2f} seconds")
print(f"\nBest regularization strength (C): {model_l2.C_[0]:.4f}")
print(f"All {X_train.shape[1]} features retained (L2 doesn't zero out features)")
print(f"\nTraining Performance:")
print(f"  Accuracy: {train_acc_l2:.4f}")
print(f"\nValidation Performance:")
print(f"  Accuracy:  {val_acc_l2:.4f}")
print(f"  Precision: {val_precision_l2:.4f}")
print(f"  Recall:    {val_recall_l2:.4f}")
print(f"  F1 Score:  {val_f1_l2:.4f}")
print(f"  AUC-ROC:   {val_auc_l2:.4f}")

results.append({
    'Model': 'L2 (Ridge)',
    'C': model_l2.C_[0],
    'Features_Selected': X_train.shape[1],  # All features
    'Train_Accuracy': train_acc_l2,
    'Val_Accuracy': val_acc_l2,
    'Val_Precision': val_precision_l2,
    'Val_Recall': val_recall_l2,
    'Val_F1': val_f1_l2,
    'Val_AUC': val_auc_l2,
    'Training_Time': l2_time
})

# Save L2 model
print(f"\nSaving L2 model to: {MODEL_L2_PATH}")
with open(MODEL_L2_PATH, 'wb') as f:
    pickle.dump(model_l2, f)
print(f"  ✓ L2 model saved")

# Model 3: ElasticNet (L1 + L2 combined) - Best of both worlds
print("\n" + "-"*80)
print("MODEL 3: ELASTICNET (L1 + L2 COMBINED)")
print("-"*80)
print("What it does:")
print("  - Combines L1 and L2 penalties")
print("  - Can select features (like L1) AND handle correlations (like L2)")
print("  - Often the most robust choice")
print()
print("Training with 5-fold cross-validation...")

elastic_start = time.time()

model_elastic = LogisticRegressionCV(
    Cs=10,
    cv=5,
    penalty='elasticnet',     # ElasticNet = L1 + L2
    solver='saga',            # Required for elasticnet
    l1_ratios=[0.5],          # 50% L1, 50% L2 (you can try different ratios)
    scoring='f1',
    max_iter=2000,            # May need more iterations
    random_state=42,
    n_jobs=-1
)

model_elastic.fit(X_train_scaled, y_train)
elastic_time = time.time() - elastic_start

# Get predictions
y_train_pred_elastic = model_elastic.predict(X_train_scaled)
y_val_pred_elastic = model_elastic.predict(X_val_scaled)
y_val_proba_elastic = model_elastic.predict_proba(X_val_scaled)[:, 1]

# Calculate metrics
train_acc_elastic = accuracy_score(y_train, y_train_pred_elastic)
val_acc_elastic = accuracy_score(y_val, y_val_pred_elastic)
val_precision_elastic = precision_score(y_val, y_val_pred_elastic)
val_recall_elastic = recall_score(y_val, y_val_pred_elastic)
val_f1_elastic = f1_score(y_val, y_val_pred_elastic)
val_auc_elastic = roc_auc_score(y_val, y_val_proba_elastic)

non_zero_coefs_elastic = np.sum(model_elastic.coef_ != 0)

print(f"\n✓ Training complete in {elastic_time:.2f} seconds")
print(f"\nBest regularization strength (C): {model_elastic.C_[0]:.4f}")
print(f"Features selected (non-zero coefficients): {non_zero_coefs_elastic}/{X_train.shape[1]}")
print(f"\nTraining Performance:")
print(f"  Accuracy: {train_acc_elastic:.4f}")
print(f"\nValidation Performance:")
print(f"  Accuracy:  {val_acc_elastic:.4f}")
print(f"  Precision: {val_precision_elastic:.4f}")
print(f"  Recall:    {val_recall_elastic:.4f}")
print(f"  F1 Score:  {val_f1_elastic:.4f}")
print(f"  AUC-ROC:   {val_auc_elastic:.4f}")

results.append({
    'Model': 'ElasticNet',
    'C': model_elastic.C_[0],
    'Features_Selected': non_zero_coefs_elastic,
    'Train_Accuracy': train_acc_elastic,
    'Val_Accuracy': val_acc_elastic,
    'Val_Precision': val_precision_elastic,
    'Val_Recall': val_recall_elastic,
    'Val_F1': val_f1_elastic,
    'Val_AUC': val_auc_elastic,
    'Training_Time': elastic_time
})

# Save ElasticNet model
print(f"\nSaving ElasticNet model to: {MODEL_ELASTIC_PATH}")
with open(MODEL_ELASTIC_PATH, 'wb') as f:
    pickle.dump(model_elastic, f)
print(f"  ✓ ElasticNet model saved")

# Step 4: Compare models
print("\n" + "="*80)
print("STEP 4: MODEL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)

print("\nComparison Table:")
print("="*100)
print(f"{'Model':<15} {'Val F1':<10} {'Val AUC':<10} {'Val Acc':<10} {'Features':<10} {'Train Time':<12}")
print("-"*100)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<15} {row['Val_F1']:<10.4f} {row['Val_AUC']:<10.4f} "
          f"{row['Val_Accuracy']:<10.4f} {row['Features_Selected']:<10} {row['Training_Time']:<12.2f}s")
print("="*100)

# Find best model
best_idx = results_df['Val_F1'].idxmax()
best_model = results_df.iloc[best_idx]

# Select the best model object
if best_model['Model'] == 'L1 (Lasso)':
    best_model_obj = model_l1
elif best_model['Model'] == 'L2 (Ridge)':
    best_model_obj = model_l2
else:
    best_model_obj = model_elastic

print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   Validation F1 Score: {best_model['Val_F1']:.4f}")
print(f"   Validation AUC-ROC:  {best_model['Val_AUC']:.4f}")
print(f"   Features Used:       {int(best_model['Features_Selected'])}/{X_train.shape[1]}")

# Save best model separately for easy access
print(f"\nSaving best model to: {BEST_MODEL_PATH}")
with open(BEST_MODEL_PATH, 'wb') as f:
    pickle.dump(best_model_obj, f)
print(f"  ✓ Best model saved (for easy loading later!)")

# Step 5: Detailed analysis of best model
print("\n" + "="*80)
print(f"STEP 5: DETAILED ANALYSIS OF BEST MODEL ({best_model['Model']})")
print("="*80)

# Get predictions for best model
if best_model['Model'] == 'L1 (Lasso)':
    y_val_pred_best = y_val_pred_l1
elif best_model['Model'] == 'L2 (Ridge)':
    y_val_pred_best = y_val_pred_l2
else:
    y_val_pred_best = y_val_pred_elastic

# Confusion Matrix
print("\nConfusion Matrix (Validation Set):")
cm = confusion_matrix(y_val, y_val_pred_best)
print(f"\n                 Predicted")
print(f"               No Spread  Spread")
print(f"Actual No      {cm[0,0]:<10} {cm[0,1]:<10}")
print(f"Actual Spread  {cm[1,0]:<10} {cm[1,1]:<10}")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_val_pred_best, 
                          target_names=['No Spread', 'Spread']))

# Feature Importance (for L1 and ElasticNet)
if best_model['Model'] in ['L1 (Lasso)', 'ElasticNet']:
    print(f"\nTop 10 Most Important Features (by absolute coefficient):")
    coefs = best_model_obj.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("-"*80)
    print(f"{'Rank':<6} {'Feature':<40} {'Coefficient':<15}")
    print("-"*80)
    for i, row in enumerate(feature_importance.head(10).iterrows(), 1):
        feat_name = row[1]['Feature']
        coef_val = row[1]['Coefficient']
        print(f"{i:<6} {feat_name:<40} {coef_val:<15.6f}")

# Step 6: Save results
print("\n" + "="*80)
print("STEP 6: SAVING RESULTS")
print("="*80)

with open(RESULTS_PATH, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LOGISTIC REGRESSION TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Training Data: {TRAINING_PATH}\n")
    f.write(f"Validation Data: {VALIDATION_PATH}\n")
    f.write(f"Number of Features: {X_train.shape[1]}\n")
    f.write(f"Training Samples: {X_train.shape[0]:,}\n")
    f.write(f"Validation Samples: {X_val.shape[0]:,}\n\n")
    
    f.write("="*80 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("="*80 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write(f"BEST MODEL: {best_model['Model']}\n")
    f.write("="*80 + "\n\n")
    f.write(f"Validation F1 Score: {best_model['Val_F1']:.4f}\n")
    f.write(f"Validation AUC-ROC:  {best_model['Val_AUC']:.4f}\n")
    f.write(f"Validation Accuracy: {best_model['Val_Accuracy']:.4f}\n")
    f.write(f"Features Used:       {int(best_model['Features_Selected'])}/{X_train.shape[1]}\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write(f"  True Negatives:  {cm[0,0]}\n")
    f.write(f"  False Positives: {cm[0,1]}\n")
    f.write(f"  False Negatives: {cm[1,0]}\n")
    f.write(f"  True Positives:  {cm[1,1]}\n")

print(f"\n✓ Results saved to: {RESULTS_PATH}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
print(f"\n✓ Trained 3 models with automatic regularization tuning")
print(f"✓ Best model: {best_model['Model']}")
print(f"✓ Best validation F1: {best_model['Val_F1']:.4f}")
print(f"✓ Results saved to: {RESULTS_PATH}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("  1. Review the results file for detailed metrics")
print("  2. The best model uses {:.0f} features out of {}".format(
    best_model['Features_Selected'], X_train.shape[1]))
print("  3. Use this model for predictions on new data")
print("  4. Consider the trade-off between F1, precision, and recall for your use case")

print("\n" + "="*80)
print("SAVED FILES")
print("="*80)
print("\nModels:")
print(f"  ✓ {MODEL_L1_PATH}")
print(f"  ✓ {MODEL_L2_PATH}")
print(f"  ✓ {MODEL_ELASTIC_PATH}")
print(f"  ✓ {BEST_MODEL_PATH} ⭐ (Use this one!)")
print(f"\nScaler:")
print(f"  ✓ {SCALER_PATH} (Required for predictions!)")
print(f"\nResults:")
print(f"  ✓ {RESULTS_PATH}")

print("\n" + "="*80)
print("HOW TO USE THE SAVED MODEL")
print("="*80)
print("""
To make predictions on new data:

import pickle
import pandas as pd

# 1. Load the scaler and model
with open('{scaler}', 'rb') as f:
    scaler = pickle.load(f)
with open('{model}', 'rb') as f:
    model = pickle.load(f)

# 2. Prepare your new data (must have same features!)
new_data = pd.read_csv('your_new_data.csv')
X_new = new_data.drop(columns=['fire_spread'])  # If it exists

# 3. Standardize using the SAME scaler
X_new_scaled = scaler.transform(X_new)

# 4. Make predictions
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]

print(f"Predicted spread: {{predictions}}")
print(f"Spread probability: {{probabilities}}")
""".format(scaler=SCALER_PATH, model=BEST_MODEL_PATH))
print("="*80)
