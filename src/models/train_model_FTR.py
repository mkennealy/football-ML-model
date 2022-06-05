import numpy as np
import pandas as pd
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score, precision_recall_curve, precision_score, accuracy_score
import lightgbm as lgbm
from lightgbm import LGBMClassifier
import shap


#LightGBM parameter tuning in optuna 
#good resource in link below as a guide
#https://programming.vip/docs/lightgbm-optuna-super-parameter-automatic-tuning-tutorial-with-code-framework.html

def objective(trial, X, y):
    # Parameter grid
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100,600),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 2, 60, step=2),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1])
        
    }
    
    # 5-fold cross validation
    kfold = 5
    folds = KFold(n_splits=kfold)
    cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()

    cv_scores = np.empty(5) 
    for n_fold, (train_idx, test_idx) in enumerate(folds.split(X, y)): 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
       # train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
       # train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, random_state=42, test_size=.1)
        
        # LGBM modeling
        model = lgbm.LGBMClassifier(objective="multiclass", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            early_stopping_rounds=100,
          #  callbacks=[
           #     LightGBMPruningCallback(trial, "binary_logloss")
           # ]
            categorical_feature=cat_cols
        )
        # model prediction 
        preds = model.predict_proba(X_test)
        # Optimization index logloss minimum
        cv_scores[n_fold] = log_loss(y_test, preds)

    return np.mean(cv_scores)

def get_optimal_parameters(X,y,n_trials):
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    study.enqueue_trial(
    {'max_depth': 5, 
                    'num_leaves': 3, 
                    'min_data_in_leaf': 30, 
                    'feature_fraction': 0.25, 
                    'lambda_l2': 95, 
                    'bagging_fraction': 0.5, 
                    'boosting_type': 'gbdt', 
                    'objective': 'multi_logloss', 
                    'random_seed': 0, 
                    'num_boost_round': 200, 
                    'learning_rate': 0.1}
    )
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=n_trials)
    
    print(f"\tBest value: {study.best_value:.5f}")
    print(f"\tBest params:")
    print("\t\t{")
    optimised_params = {}
    for key, value in study.best_params.items():
        print(f"\t\t'{key}': {value},")
        optimised_params[key] = value
    print("\t\t}")
    
    return optimised_params


def get_predicted_class_from_pred_proba(preds,threshold):
    '''
    Predicted probabilities above a given threshold are assigned class 1, otherwise they are class 0
    '''
    predicted_class = []
    for pred in preds:
        if pred >=threshold:
            predicted_class.append(1)
        else:
            predicted_class.append(0)

    return predicted_class

def train_and_predict(X_train, y_train,X_test, y_test,optimal_params):
    '''
    1. fit lightgbm model with optimised parameters on training data
    2. return predicted probabilities and predicted class of test set obserbations
    '''
    lgbm_params = optimal_params
    clf = LGBMClassifier(**lgbm_params)
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.to_list()


    clf.fit(X_train, y_train, 
            categorical_feature=cat_cols,eval_set=[(X_test, y_test)],
                eval_metric="multi_logloss",early_stopping_rounds=100)

    preds = clf.predict_proba(X_test)
    preds_class = get_predicted_class_from_pred_proba(preds[:,1],0.5)   

    return preds, preds_class

def evaluate_predictions(y_test,preds, preds_class):
    '''
    prints evaluation metrics 
    '''
    print(roc_auc_score(y_test, preds,multi_class="ovr",average = 'macro'))
    print(log_loss(y_test, preds))
    print(precision_score(y_test, preds_class))
    print(accuracy_score(y_test, preds_class))

def get_cross_validated_scores_and_shap_values(X,y,parameters):
    '''
    manual cross validation to obtain unbiased shap values of every observation
    '''

    kfold = 5
    folds = KFold(n_splits=kfold)
    cat_cols = [col for col in X.columns if str(X[col].dtype) == "category"]
    lgbm_params = parameters

    cv_scores = {"auc":[],"logloss":[],"precision":[]}
    shap_values_abs = np.zeros(X.shape)
    shap_values = np.zeros(X.shape)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y =X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

        clf = LGBMClassifier(**lgbm_params)
        
        clf.fit(train_x, train_y, early_stopping_rounds=100,verbose= 50, eval_set=[(valid_x, valid_y)],eval_metric='auc', 
                categorical_feature=cat_cols)

        preds = clf.predict_proba(valid_x)
        preds_class = get_predicted_class_from_pred_proba(preds[:,1],0.5)

        # Optimization index logloss minimum
        cv_scores["auc"].append(roc_auc_score(valid_y, preds[:,1]))
        cv_scores["logloss"].append(log_loss(valid_y, preds))
        cv_scores["precision"].append(precision_score(valid_y, preds_class))
        explainer = shap.TreeExplainer(clf)

        shap_values[valid_idx] = explainer.shap_values(X.iloc[valid_idx])[1]
        shap_values = shap_values.astype(np.float32)

    shap_importance_abs = np.abs(shap_values).mean(0)
    shap_importance = shap_values.mean(0)
    df_shaps = pd.concat([pd.DataFrame(list(X.columns)),pd.DataFrame(shap_importance_abs),pd.DataFrame(shap_importance)],axis=1)
    df_shaps.columns = ["Feature","Avg. Abs. Shap","Avg. Shap"]
    df_shaps = df_shaps.sort_values(by="Avg. Abs. Shap",ascending=False).reset_index(drop=True).iloc[:15,:]
    return cv_scores, shap_values,df_shaps
