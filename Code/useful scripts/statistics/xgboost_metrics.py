import numpy as np
import pandas as pd
import argparse
import sys
import ast
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def load_data(target: str, csv_path: str) -> tuple:
    """
    Load and preprocess data from a CSV file.
    """
    data = pd.read_csv(csv_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label').values

    # Filter if needed
    rows = data.to_numpy()
    filtered_rows, filtered_labels = [], []
    for row, label in zip(rows, labels):
        # Example filter placeholder; adjust or remove as desired
        filtered_rows.append([ast.literal_eval(cell) for cell in row])
        filtered_labels.append(label)
    X = np.array([np.array(r).flatten() for r in filtered_rows])
    y = np.array(filtered_labels)
    return X, y


def calculate_xgboost_metrics(data, labels):
    # Encode labels
    enc = LabelEncoder()
    y_enc = enc.fit_transform(labels)

    # Train/test split once
    X_train, X_test, y_train, y_test = train_test_split(
        data, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Model
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        reg_lambda=1,
        reg_alpha=0.1,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Fixed train sizes
    sizes = [250, 500, 750, 1000, 1250, 1500, 1750]
    results = { 'train_size': [], 'accuracy': [], 'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [], 'roc_auc_ovr': [] }

    for s in sizes:
        idx = np.random.RandomState(42).choice(len(X_train), size=s, replace=False)
        Xt, yt = X_train[idx], y_train[idx]
        model.fit(Xt, yt)
        yp = model.predict(X_test)
        results['train_size'].append(s)
        results['accuracy'].append(accuracy_score(y_test, yp))
        results['precision_weighted'].append(precision_score(y_test, yp, average='weighted', zero_division=0))
        results['recall_weighted'].append(recall_score(y_test, yp, average='weighted', zero_division=0))
        results['f1_weighted'].append(f1_score(y_test, yp, average='weighted', zero_division=0))
        if len(np.unique(y_test)) == 2:
            proba = model.predict_proba(X_test)[:,1]
            results['roc_auc_ovr'].append(roc_auc_score(y_test, proba))
        else:
            proba = model.predict_proba(X_test)
            results['roc_auc_ovr'].append(roc_auc_score(y_test, proba, multi_class='ovr'))

    # Print summary
    print("\n" + "="*50)
    header = "TrainSize | Accuracy | Precision_w | Recall_w | F1_w | AUC_ovr"
    print(header)
    print("-"*len(header))
    for i, s in enumerate(results['train_size']):
        print(f"{s:>9} | " + 
              f"{results['accuracy'][i]:.4f}     | " +
              f"{results['precision_weighted'][i]:.4f}       | " +
              f"{results['recall_weighted'][i]:.4f}     | " +
              f"{results['f1_weighted'][i]:.4f}  | " +
              f"{results['roc_auc_ovr'][i]:.4f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    args = parser.parse_args()
    data, labels = load_data('', args.csv_path)
    calculate_xgboost_metrics(data, labels)

if __name__ == '__main__':
    main()
