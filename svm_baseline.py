import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from utils import load_data_to_dataframe, SEED
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df_coding = load_data_to_dataframe()

    le = LabelEncoder()

    X = df_coding.text
    y = le.fit_transform(df_coding.t2_ee)

    # Define the pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    # Define the parameter space
    param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}

    # Outer loop of nested CV
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=SEED)

    # Store results
    outer_results = []

    for i, (train_idx, test_idx) in enumerate(cv_outer.split(X)):
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner loop of nested CV
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
        search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', n_jobs=1, cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best = result.best_estimator_
        y_pred = best.predict(X_test)
        #acc = accuracy_score(y_test, y_pred)
        acc = f1_score(y_test, y_pred, average='macro')
        outer_results.append(acc)
        print('%s >f1-macro=%.3f, est=%.3f, cfg=%s' % (i + 1, acc, result.best_score_, result.best_params_))

    print('  >Mean f1-macro=%.3f' % (np.round(np.mean(outer_results), 2)))

    df_results = pd.DataFrame(outer_results, columns=['Accuracy'])
    df_results.plot(kind='bar', xlabel='Run', ylabel='Accuracy')
    plt.show()
