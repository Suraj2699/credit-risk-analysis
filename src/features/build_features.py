from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def create_new_features(df):
    data = df.copy()
    data['Employment_Adjusted'] = data['Present-Employment-Since'].replace(0, 1)

    ## Creating a debt to income ratio feature.
    ## We are assuming that the average income is 1000 euros per month.
    data['Debt_to_Income_Ratio'] = data['Credit-Amount'] / (data['Employment_Adjusted']*1000)
    df['Debt_to_Income_Ratio'] = data['Debt_to_Income_Ratio']

    df['Credit_Utilization'] = data['Bank-Existing-Credits'] / (data['Savings-Account(Bonds)'] + 1)

    return df

def selecting_best_linear_features(X, y):
    # Select features based on the model characteristics, whether we use a tree-based model or a linear model.
    linear_selector = RFECV(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        n_jobs=-1
    )

    linear_selector.fit(X, y.values.ravel())
    selected_features = X.columns[linear_selector.support_].to_list()
    lineardf = X[selected_features]
    return lineardf

def selecting_best_tree_features(X, y):
    # Select features based on the model characteristics, whether we use a tree-based model or a linear model.
    tree_selector = RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        n_jobs=-1
    )

    tree_selector.fit(X, y.values.ravel())
    selected_features = X.columns[tree_selector.support_].to_list()
    treedf = X[selected_features]
    return treedf