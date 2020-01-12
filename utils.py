# identify outliers with standard deviation

import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def plot_roc_cur(fper, tper, title):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()

    
def remove_outliers(data):
    
    # calculate summary statistics

    data_mean = []
    data_std = []
    for i in range(len(data.columns)):
        data_mean.append(mean(data[data.columns[i]]))
        data_std.append(std(data[data.columns[i]]))
    
    stats = pd.DataFrame({'data_mean': data_mean,
                      'data_std': data_std,
                      'two_sigma': np.array(data_std)*2,
                      'three_sigma': np.array(data_std)*3}, index=data.columns)
    
    # identify outliers

    cut_off = np.array(stats.two_sigma)
    stats['lower'], stats['upper'] = np.array(data_mean) - cut_off, np.array(data_mean) + cut_off

    # identify outliers

    outliers_set = set()
    for feat in stats.index:
        low = stats.loc[feat, 'lower']
        up = stats.loc[feat, 'upper']
        outliers_set = outliers_set | set(data[(data[feat] < low) | (data[feat] > up)].index)

    outliers = data[data.index.isin(outliers_set)]
    number_outs = outliers.shape[0]
    
    # remove outliers

    outliers_removed = data[~data.index.isin(outliers_set)]
    number_outs_rem = outliers_removed.shape[0]

    
    return list(outliers_removed.index)


# this function need feature scaling before

def top_15_features(df):
    df = df.copy()
    y = df['TARGET']
    df.drop('TARGET', axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)
    
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]
    
    feature_importance = abs(clf.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    feat_importances = pd.DataFrame({'features': X_test.columns[sorted_idx], 'weights': feature_importance[sorted_idx]})
    feat_importances.sort_values('weights', ascending=False, inplace=True)
    A_cal_0 = list(feat_importances.features.iloc[0:14,])

    return roc_auc_score(y_test, y_scores), A_cal_0