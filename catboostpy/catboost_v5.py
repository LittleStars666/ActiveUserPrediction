import csv
import datetime
import hyperopt
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import  GridSearchCV
from data_process_v4 import processing
def predict(clf2, test_set):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    print("begin to make predictions")
    res = clf2.predict_proba(test_set.values)
    uid["proba1"] = pd.Series(res[:, 1])
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"],inplace=True)
    uid.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    uid_file = "result/uid_" + str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    active_users = uid["user_id"][:25000].unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_catboost_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])

def run():
    # print("begin to load the trainset1")
    # train_set1 = processing(trainSpan=(1,19),label=True)
    # train_set1.to_csv("data/training_ld1-19.csv", header=True, index=False)
    # train_set1 = pd.read_csv("data/training_ld1-16.csv", header=0, index_col=None)
    # print(train_set1.describe())
    # print("begin to load the trainset2")
    # train_set2 = processing(trainSpan=(5,23),label=True)
    # train_set2.to_csv("data/training_ld5-23.csv", header=True, index=False)
    # train_set2 = pd.read_csv("data/training_ld8-23.csv", header=0, index_col=None)
    # print(train_set1.describe())
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,23),label=True)
    # train_set3.to_csv("data/training_ld1-23.csv", header=True, index=False)
    # train_set3 = pd.read_csv("data/training_ld1-23.csv", header=0, index_col=None)
    # print(train_set1.describe())
    print("begin to merge the trainsets")
    # train_set = pd.concat([train_set1,train_set2,train_set3],axis=0)
    # train_set = pd.concat([train_set1,train_set2],axis=0)
    # train_set.to_csv("data/training_lm5-23.csv", header=True, index=False)
    train_set = pd.read_csv("data/training_lm15-23.csv", header=0, index_col=None)
    # del train_set1,train_set2
    # gc.collect()
    print(train_set.describe())
    keep_feature = list(set(train_set.columns.values.tolist())-set(["user_id","label"]))
    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    train_label =train_set["label"]
    train_set = train_set.drop(labels=["label","user_id"], axis=1)

    print("begin to make prediction with plain features and without tuning parameters")
    initial_params =  {
        # "verbose":2,
        "loss_function":"Logloss",
        "eval_metric":"AUC",
        "custom_metric":"AUC",
        "iterations":500,
        "random_seed":42,
        "learning_rate":0.019,
        "one_hot_max_size":2,
        "depth":6,
        "border_count":128,
        "thread_count":4,
        # "class_weights":[0.1,1.8],
        # "l2_leaf_reg":6,
        # "use_best_model":True,
        # "save_snapshot":True,
        "leaf_estimation_method":'Newton',
        # "od_type":'Iter',
        # "od_wait":20,
        "od_pval":0.0000001,
        # "used_ram_limit":1024*1024*1024*12,
        # "max_ctr_complexity":3,
        # "model_size_reg":10,
    }

    scoring = {'AUC': 'roc_auc', 'f1': "f1"}
    clf1 = GridSearchCV(CatBoostClassifier(),
                      param_grid={"iterations":[600,800,1200],
                                  "learning_rate":[0.01,0.02,0.03],
                                  "depth": [4,6,8],
                                  "leaf_estimation_method":['Newton','Gradient']},
                      scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)
    # clf1 = CatBoostClassifier(**initial_params)
    # cv_data = cv(Pool(train_set.values, train_label.values), clf1.get_params(),verbose_eval=True,nfold=5)
    # print("auc validation score :{}".format(np.max(cv_data['test-Logloss-mean'])))
    clf1.fit(train_set.values, train_label.values)
    # clf1 = CatBoostClassifier(**initial_params)
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    # clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y))
    # clf.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="map")
    # selector = RFECV(clf1, step=1, cv=3,scoring="f1",verbose=1,n_jobs=-1)
    # selector.fit(train_set.values, train_label.values)
    # train_set_new = selector.transform(train_set.values)

    bs = clf1.best_score_
    print(bs)
    bp = clf1.best_params_
    print(bp)
    #
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features of catboost")
    feature_names = train_set.columns
    feature_importances = clf1.best_estimator_.feature_importances_
    # feature_importances = clf1.feature_importances_
    print(feature_importances)
    print(feature_names)
    #
    print("load the test dataset")
    # test_set = processing(trainSpan=(12, 30), label=False)
    # test_set.to_csv("data/testing_ld12-30.csv",header=True,index=False)
    test_set = pd.read_csv("data/testing_ld15-30.csv",header=0,index_col=None)
    print("begin to make prediction")
    predict(clf1,test_set)


    with open("kuaishou_stats.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["feature importance of catboost for tencent-crt ", str_time])
        writer.writerow(["best score",bs,"best params"])
        for key, value in bp.items():
            writer.writerow([key, value])
        # writer.writerow(eval_metrics)
        feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
        for score, name in feature_score_name:
            print('{}: {}'.format(name, score))
            writer.writerow([name, score])
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    model_name = "catboost_" + str_time + ".pkl"
    joblib.dump(clf1, model_name)

    print("begin to tune the parameters with the selected feature")
    paramsSpace = {
        "n_estimators":hyperopt.hp.quniform("n_estimators", 200, 2000, 100),
        'depth': hyperopt.hp.quniform("depth", 3, 8, 1),
        "border_count": hyperopt.hp.quniform("border_count", 128, 200, 6),
        'learning_rate': hyperopt.hp.loguniform('learning_rate', 1e-6, 1e-1),
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 1, 10,1),
        'bagging_temperature': hyperopt.hp.uniform('bagging_temperature', 0.6, 1.0),
        'rsm': hyperopt.hp.uniform('rsm', 0.6, 1.0),
        # "leaf_estimation_method": hyperopt.hp.choice("leaf_estimation_method",['Newton', 'Gradient']),
    }

    # train_x, val_x, train_y, val_y = train_test_split(train_set.values, train_label.values, test_size=0.33,
    #                                                   random_state=42)
    def hyperopt_objective(params):
        model = CatBoostClassifier(
            n_estimators=params["n_estimators"],
            # use_best_model=True,od_type='Iter',od_wait=20,
            verbose=1,eval_metric='AUC',
            od_pval=0.0000001,
            # leaf_estimation_method=params['leaf_estimation_method'],
            depth=params['depth'],
            border_count=params['border_count'],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params['l2_leaf_reg'],bagging_temperature=params['bagging_temperature'],
            rsm=params['rsm'])
        cv_data = cv(Pool(train_set,train_label),model.get_params(),nfold=4,verbose_eval=True)
        # model.fit(train_pool_tp, eval_set=validate_pool_tp)
        # model.fit(X=train_x, y=train_y,
        #         eval_set=(val_x, val_y))
        # y_val_hat = model.predict(train_set.values)
        # mean_auc = roc_auc_score(train_label.values, y_val_hat)
        # metrics = model.eval_metrics(validate_pool_tf, ['AUC'])
        # mean_auc = sum(metrics['AUC'])/float(len(metrics['AUC']))
        # cv_data = cv(
        #     Pool(train_set_tf, train_label, cat_features=categorical_features_indices_tf),
        #     model.get_params()
        # )
        mean_auc = np.max(cv_data['test-Logloss-mean'])
        return 1 - mean_auc  # as hyperopt minimises
    best_params = hyperopt.fmin(
        hyperopt_objective,
        space=paramsSpace,
        algo=hyperopt.tpe.suggest,
        max_evals=100,
    )
    print(best_params)
    clf2 = CatBoostClassifier(
        verbose=2,loss_function="Logloss",
        iterations=best_params["n_estimators"],
        eval_metric="AUC",
        custom_metric="AUC",
        random_seed=42,
        # use_best_model=True,
        # od_type='Iter',od_wait=20,
        # leaf_estimation_method=best_params['leaf_estimation_method'],
        depth=best_params['depth'],
        border_count=best_params['border_count'],
        learning_rate=best_params["learning_rate"],
        # l2_leaf_reg=best_params['l2_leaf_reg'],
        bagging_temperature=best_params['bagging_temperature'],rsm=best_params['rsm'])
    # cv_data = cv(Pool(train_set.values, train_label.values), clf2.get_params(),nfold=5)
    # clf2.fit(X=train_x, y=train_y,
    #           eval_set=(val_x, val_y))
    # print(cv_data)
    clf2.fit(train_set.values,train_label.values)
    print("parameter tuning over, begin to save the model!")
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    model_name = "catboost_" + str_time + ".pkl"
    clf2.save_model(model_name)
    # joblib.dump(clf2, model_name)

    print("begin to process the whole dataset and ready to feed into the fitted model")
    predict(clf2,test_set)

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = clf2.feature_importances_
    print(feature_importances)
    print(feature_names)

    with open("kuaishou_stats.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["feature importance of catboost for kuaishou-crt", str_time])
        # writer.writerow(eval_metrics)
        feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
        for score, name in feature_score_name:
            print('{}: {}'.format(name, score))
            writer.writerow([name, score])
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)
if __name__=="__main__":
    run()