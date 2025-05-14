
# Now import your modules
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint


def train_logistic_regression(X, y):
    """Training and tuning Logistic Regression model"""
    params = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2', 'l1', 'elasticnet'],
        'classifier__class_weight': [None, 'balanced']
    }

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_naive_bayes(X, y):
    """Train and tune Naive Bayes model"""

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', GaussianNB())
    ])

    param_grid = {
        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        'classifier__priors': [None, [0.7, 0.3], [0.3, 0.7], [0.5, 0.5]]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        random_state=42
    )

    grid_search.fit(X, y)

    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_decision_tree(X, y):
    """Train and tune Decision Tree model"""
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None] + list(range(5, 30)),
        'classifier__min_samples_split': randint(2, 20)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use RandomizedSearchCV for faster tuning
    
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_random_forest(X, y):
    """Train and tune Random Forest model"""
    # Define the pipeline with SMOTE and RandomForestClassifier
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__max_depth': [None] + list(range(5, 30)),
        'classifier__n_estimators': randint(100, 1000),
        'classifier__min_samples_split': randint(2, 20),
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__bootstrap': [True, False]
    }
    
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_xgboost(X, y):
    """Train and tune XGBoost model"""

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    param_grid = {
        'classifier__max_depth': [None] + list(range(5, 30)),
        'classifier__n_estimators': randint(100, 1000),
        'classifier__learning_rate': [0.1, 0.01, 0.05],
        'classifier__colsample_bytree': [0.2, 0.5, 0.8, 1]
    }

    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_lightgbm(X, y):
    """Train and tune LightGBM model"""
    # Define the pipeline with SMOTE and LGBMClassifier

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LGBMClassifier())
    ])

    # Define the parameter grid for RandomizedSearchCV
    # Note: We can adjust the ranges and values based on your needs

    param_grid = {
        'classifier__max_depth': [None] + list(range(5, 30)),
        'classifier__n_estimators': randint(100, 1000),
        'classifier__learning_rate': [0.1, 0.01, 0.05],
        'classifier__colsample_bytree': [0.2, 0.5, 0.8, 1],
        "classifier__num_leaves": randint(10, 150),
        "classifier__subsample": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score

def train_catboost(X, y):
    """Train and tune CatBoost model"""

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', CatBoostClassifier(silent=True))
    ])

    param_grid = {
        'classifier__depth': list(range(1, 16)),
        'classifier__iterations': randint(100, 1000),
        'classifier__learning_rate': [0.1, 0.01, 0.05],
        'classifier__colsample_bylevel': [0.2, 0.5, 0.8, 1],
        'classifier__l2_leaf_reg': randint(1, 10),
        'classifier__border_count': [4, 8, 16, 32, 64, 128],
        'classifier__random_strength': randint(1, 10)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    # Return the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score