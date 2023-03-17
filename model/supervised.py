# -*- coding: utf-8 -*-
'''
Created on Mon Nov  8 16:01:33 2021

@author: shangfr
'''

import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_pinball_loss, mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,  precision_recall_curve


def model_cls(datasets, model_parm):
    '''Classifier.
    '''
    X = datasets['X']
    y = datasets['y']
    feature_names = datasets['feature_names']
    target_names = datasets['target_names']
    pos_id = datasets['pos_id']

    score_criterion = model_parm['score_criterion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # 步骤二：定义模型，创建管道Pipeline
    clf = GradientBoostingClassifier()

    # 步骤三：pipeline、自动寻参和交叉验证的组合使用
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='micro', labels=[pos_id], zero_division=1),
               'recall': make_scorer(recall_score, average='macro', labels=[pos_id], zero_division=1)}

    param_grid = {'max_depth': [3, 5, 8],
                  'max_features': ['log2', 'sqrt', None]}

    rand_model = RandomizedSearchCV(
        clf,  param_grid, scoring=scoring[score_criterion], random_state=0)

    search = rand_model.fit(X_train, y_train)
    # search.best_params_
    best_model = rand_model.best_estimator_

    # Classification metrics
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    cls_report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=1)

    y_score = best_model.decision_function(X_test)

    if len(y_score.shape) > 1:
        y_score = y_score[:, pos_id]

    fpr, tpr, thresholds = roc_curve(
        y_test, y_score, pos_label=best_model.classes_[pos_id])
    prec, recall, _ = precision_recall_curve(
        y_test, y_score, pos_label=best_model.classes_[pos_id])

    AP = np.sum((recall[:-1] - recall[1:]) * prec[:-1]).round(2)

    result =f'''
            ### :blue[classification metrics]
            🔴 **AUC**: {auc(fpr, tpr):.2f}  🔴 **AP**: {AP:.2f}
            
            ---
            '''
            
    report = {'score': result}

    feature_importance = best_model.feature_importances_
    ind = np.argsort(-feature_importance)
    feature_names = [feature_names[i] for i in ind]

    fig_data = {'feature_importance': {'names': feature_names, 'importance': feature_importance[ind].tolist()},
                'cls_report':'>' + cls_report,
                'cm': {'data': cm.tolist(), 'classes': target_names, 'title':'Confusion Matrix'},
                'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'AUC': round(auc(fpr, tpr), 2), 'positive': target_names[pos_id]},
                'pr': {'prec': prec.tolist(), 'recall': recall.tolist(), 'AP': AP}
                }

    return report, fig_data, best_model


def model_regr(datasets, model_parm):
    '''Regressor.
    '''
    X = datasets['X']
    y = datasets['y']
    feature_names = datasets['feature_names']
    score_criterion = model_parm['score_criterion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    # 步骤二：定义模型，创建管道Pipeline
    regr = GradientBoostingRegressor()

    # 步骤三：pipeline、自动寻参和交叉验证的组合使用
    scoring = {'mean_squared_error': make_scorer(mean_squared_error),
               'mean_pinball_loss': make_scorer(mean_pinball_loss, greater_is_better=False)}

    param_grid = {'max_depth': [3, 5, 8],
                  'max_features': ['log2', 'sqrt', None]}
    
    sk_model_rand = RandomizedSearchCV(
        regr,  param_grid, scoring=scoring[score_criterion], random_state=0)
    search = sk_model_rand.fit(X_train, y_train)
    # search.best_params_
    best_sk_model = sk_model_rand.best_estimator_
    feature_importance = best_sk_model.feature_importances_
    ind = np.argsort(-feature_importance)

    y_pred = best_sk_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result =f'''
            ### :blue[Regression metrics] 
            🔴 **MSE**: {round(mse,3)}  🔴 **R2**: {round(r2,3)} 
            
            ---
            '''
    report = {'score': result}
    feature_names = [feature_names[i] for i in ind]
    fig_data = {'feature_importance': {'names': feature_names, 'importance': feature_importance[ind].round(3).tolist()},
                'y_vs':{'y_true':y_test.round(3).tolist(),'y_pred':y_pred.round(3).tolist()}
                }

    # df=pd.DataFrame.from_dict(search.cv_results_)
    return report, fig_data, best_sk_model
