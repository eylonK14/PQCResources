# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import ast
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

def load_data_with_label_split(csv_path: str, train_labels: list, test_labels: list, test_size: float = 0.2):
    """
    Load and preprocess data with different labels for training and testing.
    
    :param csv_path: Path to the CSV file to load.
    :param train_labels: List of labels to include in training (e.g., [1, 2, 3])
    :param test_labels: List of labels to include in testing (e.g., [1, 2])
    :param test_size: Proportion of data to use for testing
    :return: Tuple (X_train, X_test, y_train, y_test)
    """
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV at '{csv_path}': {e}")
        return None, None, None, None
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    labels = data.pop('label').values
    
    # Log original label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Original label distribution:", dict(zip(unique, counts)))
    
    # Process the data (assuming similar structure to your original code)
    rows = data.to_numpy()
    processed_rows = []
    
    for row in rows:
        processed_rows.append([ast.literal_eval(cell) for cell in row])
    
    X = np.array([np.array(r).flatten() for r in processed_rows])
    y = labels
    
    # Create train/test split ensuring we have the right labels in each set
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Filter training data to include only specified train_labels
    train_mask = np.isin(y_train_full, train_labels)
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    
    # Filter test data to include only specified test_labels
    test_mask = np.isin(y_test_full, test_labels)
    X_test = X_test_full[test_mask]
    y_test = y_test_full[test_mask]
    
    print(f"Training samples: {len(y_train)} with labels {np.unique(y_train)}")
    print(f"Testing samples: {len(y_test)} with labels {np.unique(y_test)}")
    
    # Log final distributions
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("Training label distribution:", dict(zip(unique_train, counts_train)))
    print("Testing label distribution:", dict(zip(unique_test, counts_test)))
    
    return X_train, X_test, y_train, y_test

def evaluate_model_with_custom_split(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train model on training data and evaluate on test data.
    """
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)  # Note: this will only work if test labels are subset of train labels
    
    # Train the model
    model.fit(X_train, y_train_encoded)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted')
    recall = recall_score(y_test_encoded, y_pred, average='weighted')
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    
    # Calculate AUC (only for binary classification or if predict_proba is available)
    try:
        if y_pred_proba is not None and len(np.unique(y_test_encoded)) == 2:
            auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        elif y_pred_proba is not None:
            auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        else:
            auc = np.nan
    except:
        auc = np.nan
    
    return {
        'Model': model_name,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-score': f"{f1:.4f}",
        'AUC': f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    }

def run_models_with_label_mismatch(csv_path: str, train_labels: list, test_labels: list, 
                                 output_suffix: str = "label_mismatch"):
    """
    Runs multiple classification models with different labels for training and testing.
    
    Args:
        csv_path (str): Path to the CSV file containing the data.
        train_labels (list): Labels to include in training data (e.g., [1, 2, 3]).
        test_labels (list): Labels to include in testing data (e.g., [1, 2]).
        output_suffix (str): Suffix for the output CSV file.
    
    Returns:
        pd.DataFrame: Results dataframe with model performance metrics.
    """
    # Load and split data
    X_train, X_test, y_train, y_test = load_data_with_label_split(
        csv_path, train_labels, test_labels
    )
    
    if X_train is None:
        return None
    
    # Define models
    models = [
        RandomForestClassifier(random_state=42),
        XGBClassifier(random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42, max_iter=1000),
        GaussianNB(),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)
    ]
    
    model_names = [
        'Random Forest',
        'XGBoost',
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'MLP',
        'Naive Bayes',
        'AdaBoost',
        'Gradient Boosting'
    ]
    
    # Sort models and names
    model_names, models = zip(*sorted(zip(model_names, models)))
    
    # Evaluate each model
    results = []
    for model, name in zip(models, model_names):
        try:
            result = evaluate_model_with_custom_split(
                model, X_train, X_test, y_train, y_test, name
            )
            results.append(result)
            print(f"Completed: {name}")
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_filename = f'results_{output_suffix}.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
    
    return results_df

def run_models_with_stratified_10fold(csv_path: str, train_labels: list, test_labels: list, 
                                    output_suffix: str = "stratified_10fold"):
    """
    Runs models using stratified 10-fold cross-validation where each fold 
    trains on train_labels but validates only on test_labels.
    
    The stratification is done on the full dataset, but training/testing 
    is filtered according to the specified labels.
    """
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    labels = data.pop('label').values
    
    # Log original label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Original label distribution:", dict(zip(unique, counts)))
    
    # Process data (same as your original preprocessing)
    rows = data.to_numpy()
    processed_rows = []
    for row in rows:
        processed_rows.append([ast.literal_eval(cell) for cell in row])
    
    X = np.array([np.array(r).flatten() for r in processed_rows])
    y = labels
    
    # Define models
    models = [
        RandomForestClassifier(random_state=42),
        XGBClassifier(random_state=42),
        LogisticRegression(random_state=42, max_iter=1000),
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42, max_iter=1000),
        GaussianNB(),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)
    ]
    
    model_names = [
        'Random Forest', 'XGBoost', 'Logistic Regression', 'KNN',
        'Decision Tree', 'MLP', 'Naive Bayes', 'AdaBoost', 'Gradient Boosting'
    ]
    
    # Sort models and names (matching your original code style)
    model_names, models = zip(*sorted(zip(model_names, models)))
    
    # Prepare results storage (matching your original format)
    results = pd.DataFrame(
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    )
    
    print(f"Using Stratified 10-Fold CV: Train on {train_labels}, Test on {test_labels}")
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        fold_scores = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
        }
        
        # Stratified 10-fold cross-validation on the full dataset
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_num = 0
        for train_idx, val_idx in skf.split(X, y):
            fold_num += 1
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Filter training data to include only train_labels
            train_mask = np.isin(y_fold_train, train_labels)
            X_train_filtered = X_fold_train[train_mask]
            y_train_filtered = y_fold_train[train_mask]
            
            # Filter validation data to include only test_labels
            val_mask = np.isin(y_fold_val, test_labels)
            X_val_filtered = X_fold_val[val_mask]
            y_val_filtered = y_fold_val[val_mask]
            
            # Skip fold if we don't have enough samples
            if len(X_train_filtered) == 0 or len(X_val_filtered) == 0:
                print(f"  Fold {fold_num}: Skipped (insufficient samples)")
                continue
                
            # Check if we have all train_labels in training set for this fold
            unique_train_labels = np.unique(y_train_filtered)
            if not all(label in unique_train_labels for label in train_labels):
                print(f"  Fold {fold_num}: Missing some train labels, continuing anyway...")
            
            # Check if we have all test_labels in validation set for this fold
            unique_val_labels = np.unique(y_val_filtered)
            if not all(label in unique_val_labels for label in test_labels):
                print(f"  Fold {fold_num}: Missing some test labels in validation set")
            
            try:
                # Encode labels - fit on training labels, transform both sets
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train_filtered)
                
                # For test labels, we need to be careful about unseen labels
                # Create a mapping for test labels that exist in training
                test_label_mapping = {}
                for i, label in enumerate(le.classes_):
                    if label in test_labels:
                        test_label_mapping[label] = i
                
                # Only keep test samples whose labels exist in training set
                valid_test_mask = np.isin(y_val_filtered, list(test_label_mapping.keys()))
                X_val_final = X_val_filtered[valid_test_mask]
                y_val_final = y_val_filtered[valid_test_mask]
                
                if len(y_val_final) == 0:
                    print(f"  Fold {fold_num}: No valid test samples after filtering")
                    continue
                
                y_val_encoded = le.transform(y_val_final)
                
                # Train and evaluate
                model.fit(X_train_filtered, y_train_encoded)
                y_pred = model.predict(X_val_final)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val_encoded, y_pred)
                
                # Handle potential division by zero in precision/recall/f1
                try:
                    precision = precision_score(y_val_encoded, y_pred, average='weighted')
                except:
                    precision = 0.0
                
                try:
                    recall = recall_score(y_val_encoded, y_pred, average='weighted')
                except:
                    recall = 0.0
                
                try:
                    f1 = f1_score(y_val_encoded, y_pred, average='weighted')
                except:
                    f1 = 0.0
                
                # Calculate AUC
                try:
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_val_final)
                        if len(np.unique(y_val_encoded)) == 2:
                            # Binary classification
                            auc = roc_auc_score(y_val_encoded, y_pred_proba[:, 1])
                        else:
                            # Multi-class classification
                            auc = roc_auc_score(y_val_encoded, y_pred_proba, multi_class='ovr')
                    else:
                        auc = np.nan
                except Exception as auc_error:
                    print(f"  Fold {fold_num}: AUC calculation failed: {auc_error}")
                    auc = np.nan
                
                # Store scores
                fold_scores['accuracy'].append(accuracy)
                fold_scores['precision'].append(precision)
                fold_scores['recall'].append(recall)
                fold_scores['f1'].append(f1)
                fold_scores['auc'].append(auc)
                
                print(f"  Fold {fold_num}: Acc={accuracy:.3f}, F1={f1:.3f} "
                      f"(Train: {len(y_train_filtered)}, Test: {len(y_val_final)})")
                
            except Exception as e:
                print(f"  Fold {fold_num}: Error - {e}")
                continue
        
        # Calculate mean and std for each metric (matching your original format)
        if fold_scores['accuracy']:  # Check if we have any scores
            acc_mean, acc_std = np.mean(fold_scores['accuracy']), np.std(fold_scores['accuracy'])
            prec_mean, prec_std = np.mean(fold_scores['precision']), np.std(fold_scores['precision'])
            rec_mean, rec_std = np.mean(fold_scores['recall']), np.std(fold_scores['recall'])
            f1_mean, f1_std = np.mean(fold_scores['f1']), np.std(fold_scores['f1'])
            
            # Handle AUC (might have NaN values)
            auc_values = [x for x in fold_scores['auc'] if not np.isnan(x)]
            if auc_values:
                auc_mean, auc_std = np.mean(auc_values), np.std(auc_values)
                auc_result = f"{auc_mean:.2f} +/- {auc_std:.2f}"
            else:
                auc_result = "N/A"
            
            # Format results to match your original style
            results.loc[len(results)] = [
                name,
                f"{acc_mean:.2f} +/- {acc_std:.2f}",
                f"{prec_mean:.2f} +/- {prec_std:.2f}",
                f"{rec_mean:.2f} +/- {rec_std:.2f}",
                f"{f1_mean:.2f} +/- {f1_std:.2f}",
                auc_result
            ]
            
            print(f"Completed {name}: {len(fold_scores['accuracy'])} successful folds")
        else:
            print(f"No valid folds for {name}")
    
    # Save results
    output_filename = f'results_{output_suffix}.csv'
    results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Set your parameters
    csv_path = 'C:\\Users\\Eylon\\PQC\\JournalDatasets\\pqc-algo-multi.csv'  # Replace with your CSV file path
    train_labels = [0, 1, 2]  # Labels to include in training
    test_labels = [1, 2]      # Labels to include in testing
    
    print("Running models with train/test split approach...")
    results1 = run_models_with_label_mismatch(
        csv_path, train_labels, test_labels, "train_012_test_12"
    )
    
    print("\nRunning models with STRATIFIED 10-FOLD cross-validation...")
    results2 = run_models_with_stratified_10fold(
        csv_path, train_labels, test_labels, "stratified_10fold_012_12"
    )
    
    if results1 is not None:
        print("\nTrain/Test Split Results:")
        print(results1)
    
    if results2 is not None:
        print("\nStratified 10-Fold CV Results:")
        print(results2)