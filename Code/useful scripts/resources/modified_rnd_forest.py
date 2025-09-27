#!/usr/bin/env python3

import ast
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report


def load_data(args):
    data = pd.read_csv('tdl/all_files.csv')
    data = data.drop(columns=['Unnamed: 0'])
    labels = data.pop('label')
    data = data.to_numpy()
    labels = labels.iloc[:].values
    new_data = []
    for tdl, label in zip(data, labels):
        if args.browser or args.operating_system or args.all:
            if label % 2 == 0:
                continue
        new_data.append([ast.literal_eval(j) for j in tdl])
    arg = 'pqc' if args.pqc else 'browser' if args.browser else 'os' if args.operating_system else 'all' if args.all else 'everything'
    if args.pqc:
        labels = np.array([i % 2 for i in labels])
    elif args.browser:
        labels = np.array([((i // 10) % 10) * 10 for i in labels if i % 2 == 1])
    elif args.operating_system:
        labels = np.array([(i // 100) * 100 for i in labels if i % 2 == 1])
    elif args.all:
        labels = np.array([i for i in labels if i % 2 == 1])
    data = np.array(new_data)
    data = np.array([i.flatten() for i in data])

    return data, labels, arg


def train_model(args):
    # params for the random forest. Pretty basic. Could also use GridSearchCV
    # or RandomizedSearchCV to find the best params more cleanly
    # num_packets_options = [10, 20, 30, 50, 100, 200]
    n_estimators_options = [10, 50, 100, 200]
    max_depth_options = [3, 5, 10]

    results = []

    # for num_packets in num_packets_options:
    data, labels, arg = load_data(args)

    print(f'{data=}\n{labels=}')

    # 70% of the data is used for training, 30% for testing
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.3,
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    max_score = 0
    best_params = None
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            scores = cross_val_score(clf, x_train, y_train, cv=skf)
            mean_score = scores.mean()
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'mean_accuracy': mean_score
            })
            if mean_score > max_score:
                max_score = mean_score
                best_params = (n_estimators, max_depth)
            print(f'{n_estimators=}, {max_depth=}, {mean_score=:.2f}')

    clf = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        random_state=42
    )
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(f'{predictions=}')
    report = classification_report(y_test, predictions, output_dict=True)

    report.update({
        "accuracy": {
            "precision": None,
            "recall": None,
            "f1-score": report["accuracy"],
            "support": report['macro avg']['support']
        }
    })

    results_df = pd.DataFrame(report).transpose()
    results_df.to_csv(f'random_forest_results_{arg}.csv', index=True)

    print(f'{best_params=}, {max_score=:.2f}')


def main():
    parser = argparse.ArgumentParser(
        prog='Random Forest Classifier',
        description='Random Forest Classifier for TDL',
        epilog='Enjoy the program! :)')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-p',
        '--pqc',
        action='store_true',
        help='PQC classifier'
    )
    group.add_argument(
        '-b',
        '--browser',
        action='store_true',
        help='using pqc: Browser classifier'
    )
    group.add_argument(
        '-os',
        '--operating_system',
        action='store_true',
        help='using pqc: OS classifier'
    )
    group.add_argument(
        '-a',
        '--all',
        action='store_true',
        help='using pqc: All classifiers'
    )
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
