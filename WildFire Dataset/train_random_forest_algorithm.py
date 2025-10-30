import pandas as pd
import argparse
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer # Corrected import
import joblib

def load_and_split(file_path, sep=','):
    """Loads a CSV and splits it into features (X) and target (y)."""
    print(f"\n  - Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep=sep)
        print(f"    - Successfully loaded data. Shape: {df.shape}")
        print(f"    - Columns: {list(df.columns)}")
        print(f"    - Sample data head:\n{df.head()}")
    except Exception as e:
        print(f"  [ERROR] Failed to read file: {e}")
        print(f"  Please check your file path and --sep argument.")
        return None, None

    if 'fire_spread' in df.columns:
        X = df.drop('fire_spread', axis=1)
        y = df['fire_spread']
        print("    - 'fire_spread' column found and used as target.")
    else:
        # Fallback if target column is just the last one
        print("    - 'fire_spread' column not found.")
        if df.shape[1] > 1:
            print("    - Assuming last column is target.")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print(f"    - Features shape: {X.shape}, Target shape: {y.shape}")
        else:
            print("    - [ERROR] Cannot determine features and target. DataFrame has only one column.")
            return None, None

    return X, y

def train_model(train_files, output_dir, sep):
    """
    Trains a high-performance RF model, tunes it, and saves the final
    model and preprocessor.
    """
    script_start_time = time.time()

    # --- Section 1: What is Random Forest? ---
    print("\n" + "="*80)
    print(" SCRIPT 1: TRAINING THE MODEL")
    print("="*80)
    print(" This script will train the best possible Random Forest model.")
    print(" It will combine all training/validation files, find the best")
    print(" hyperparameters, and then save the final, tuned model.")

    # --- Section 2: Loading & Preparing Training Data ---
    print("\n" + "="*80)
    print(" SECTION 2: LOADING & PREPARING TRAINING DATA")
    print("="*80)
    print("\n [2.1] Loading Training Files:")
    X_train_list, y_train_list = [], []
    for f in train_files:
        X, y = load_and_split(f, sep=sep)
        if X is None:
            print(f"[FATAL ERROR] Could not load training data from {f}. Exiting.")
            return
        X_train_list.append(X)
        y_train_list.append(y)

    # Check if any data was loaded
    if not X_train_list:
        print("[FATAL ERROR] No training data loaded from any file. Exiting.")
        return

    print("\n [2.1.1] Concatenating DataFrames:")
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    print(f"\n   Total training samples loaded: {len(y_train)}")
    print(f"   Combined Features Shape: {X_train.shape}")
    print(f"   Combined Target Shape: {y_train.shape}")
    print(f"   Combined Features Head:\n{X_train.head()}")


    print("\n [2.2] Preprocessing: Fitting the Imputer")
    print("   Applying 'median imputation' to learn the data's structure.")
    imputer = SimpleImputer(strategy='median')

    # --- Debugging: Print data types ---
    print("\n   Data types before imputation:")
    print(X_train.dtypes)
    # --- End Debugging ---

    # Ensure X_train is not empty before fitting imputer
    if X_train.empty:
        print("[FATAL ERROR] X_train is empty after concatenation. Cannot fit imputer. Exiting.")
        return


    train_cols = X_train.columns
    # Check if all columns are non-numeric before imputation
    if not any(pd.api.types.is_numeric_dtype(X_train[col]) for col in train_cols):
        print("[FATAL ERROR] All columns in X_train are non-numeric. SimpleImputer requires at least one numeric column. Exiting.")
        print("Please check your data files to ensure they contain numeric features.")
        return


    X_train = imputer.fit_transform(X_train) # Use fit_transform here
    X_train = pd.DataFrame(X_train, columns=train_cols)
    print("   Imputer has been fitted.")

    # --- Section 3: Hyperparameter Tuning (GridSearchCV) ---
    print("\n" + "="*80)
    print(" SECTION 3: FINDING THE BEST MODEL (TUNING)")
    print("="*80)
    print("   Using GridSearchCV to test many combinations of settings.")
    print("   This is the most time-consuming part.")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='f1')

    print("\n   Starting GridSearch... (This will take a long time)\n")
    tune_start_time = time.time()
    grid_search.fit(X_train, y_train)
    tune_time = time.time() - tune_start_time
    print(f"\n   Tuning complete in {tune_time / 60:.2f} minutes.")

    best_rf_model = grid_search.best_estimator_
    print("\n   The best model has been found. ✅")
    print(f"   Best settings found: {grid_search.best_params_}")

    # --- Section 4: Saving Model and Artifacts ---
    print("\n" + "="*80)
    print(f" SECTION 4: SAVING MODEL TO '{output_dir}' FOLDER")
    print("="*80)
    os.makedirs(output_dir, exist_ok=True)

    # Save the best model
    model_path = os.path.join(output_dir, 'rf_model.joblib')
    joblib.dump(best_rf_model, model_path)
    print(f"   Final model saved to: {model_path}")

    # Save the imputer (CRITICAL for testing)
    imputer_path = os.path.join(output_dir, 'imputer.joblib')
    joblib.dump(imputer, imputer_path)
    print(f"   Data imputer saved to: {imputer_path}")

    # Save the best parameters to a text file
    params_path = os.path.join(output_dir, 'best_params.txt')
    with open(params_path, 'w') as f:
        f.write("Best Hyperparameters Found:\n")
        f.write(str(grid_search.best_params_))
    print(f"   Best parameters saved to: {params_path}")

    print("\n" + "="*80)
    print(f"--- SCRIPT 1 FINISHED (Total Time: {(time.time() - script_start_time) / 60:.2f} mins) ---")
    print("="*80)

if __name__ == '__main__':
    # Define hard-coded file paths
    train_files = [
        "/home/boa/Wildfire-Dataset/WildFire Dataset/features_array_training_set.csv",
        "/home/boa/Wildfire-Dataset/WildFire Dataset/features_array_validation_set.csv"
    ]
    # Save trained artifacts under the existing repo folder structure
    output_dir = "/home/boa/Wildfire-Dataset/WildFire Dataset/trained RF models/Training"
    # Files appear to be TSV (tab-delimited)
    sep = '\t'

    train_model(train_files, output_dir, sep)