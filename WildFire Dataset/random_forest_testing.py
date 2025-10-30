import pandas as pd
import argparse
import os
import time
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import joblib

def load_and_split(file_path, sep='\t'):
    """Loads a CSV and splits it into features (X) and target (y)."""
    print(f"  - Loading data from: {file_path}")
    df = pd.read_csv(file_path, sep=sep)
    
    if 'fire_spread' in df.columns:
        X = df.drop('fire_spread', axis=1)
        y = df['fire_spread']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
    return X, y

def test_model(test_file, model_dir, sep='\t'):
    """
    Loads a saved model and imputer, runs them on the test set,
    and prints the final performance.
    """
    script_start_time = time.time()
    
    # --- Section 1: Loading Model and Test Data ---
    print("\n" + "="*80)
    print(" SCRIPT 2: TESTING THE MODEL")
    print("="*80)
    print(f" Loading model and artifacts from: {model_dir}")

    # Load the saved model and imputer
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    imputer_path = os.path.join(model_dir, 'imputer.joblib')
    
    try:
        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
    except FileNotFoundError:
        print(f"\n[ERROR] Model or imputer not found in '{model_dir}'.")
        print("Please run the 'train_model.py' script first.")
        return

    print("   Model and imputer loaded successfully. ✅")

    print("\n [1.1] Loading Test File (Final Exam):")
    X_test, y_test = load_and_split(test_file, sep=sep)
    print(f"   Total testing samples loaded: {len(y_test)}")
    test_cols = X_test.columns # Save column names

    # --- Section 2: Preprocessing Test Data ---
    print("\n" + "="*80)
    print(" SECTION 2: PREPARING THE TEST DATA")
    print("="*80)
    print("   Applying the *saved* imputer rules to the test data.")
    # Note: We use .transform() ONLY. We are not re-learning.
    X_test = imputer.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=test_cols)
    print("   Test data is clean and ready.")

    # --- Section 3: Making Final Predictions ---
    print("\n" + "="*80)
    print(" SECTION 3: RUNNING FINAL PREDICTIONS")
    print("="*80)
    print("   Running the 'final exam'...")
    y_pred = model.predict(X_test)
    print("   Predictions complete. ✅")

    # --- Section 4: Final Performance & Saving Report ---
    print("\n" + "="*80)
    print(" SECTION 4: FINAL PERFORMANCE RESULTS")
    print("="*80)
    
    # --- 4a. Print Performance to Terminal ---
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) # F1 for the positive class '1'
    report = classification_report(y_test, y_pred, target_names=['No Spread (0)', 'Spread (1)'])
    
    print("\n--- Key Performance Metrics ---")
    print(f"  Accuracy: {accuracy: .4f}")
    print(f"  F1-Score (for 'Spread'): {f1: .4f}")
    
    print("\n--- Full Classification Report ---")
    print(report)
    
    # --- 4b. Save Report to Model Folder ---
    print("\n [4.1] Saving Final Report...")
    report_path = os.path.join(model_dir, 'test_performance_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80)
        f.write("\n--- Random Forest FINAL TEST Performance ---\n")
        f.write("="*80)
        f.write(f"\nTest performed at: {time.ctime()}\n")
        f.write("\n--- Key Metrics ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Spread): {f1:.4f}\n")
        f.write("\n--- Full Classification Report ---\n")
        f.write(report)
    
    print(f"   Detailed performance report saved to: {report_path}")
    print("\n" + "="*80)
    print(f"--- SCRIPT 2 FINISHED (Total Time: {(time.time() - script_start_time):.2f} seconds) ---")
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a saved RF model.')

    parser.add_argument(
        '--test_file',
        type=str,
        default='/home/boa/Wildfire-Dataset/WildFire Dataset/features_array_testing_set.csv',
        help='Path to the final test data CSV file.'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/boa/Wildfire-Dataset/WildFire Dataset/trained RF models/Training',
        help="Directory to load the model and imputer from."
    )

    args = parser.parse_args()

    test_model(args.test_file, args.model_dir, sep='\t')