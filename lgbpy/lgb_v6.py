import csv
import datetime
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import  GridSearchCV
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from data_process_v3 import processing
from skopt.space import Categorical

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
    active_users = uid["user_id"][:23700].unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_lgb_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])
# using this module ,one needs to deconstruct some of the features in data_process
def run():
    print("begin to load the trainset1")
    # train_set1 = processing(trainSpan=(1,16),label=True)
    # train_set1.to_csv("data/training_d1-11.csv", header=True, index=False)
    train_set1 = pd.read_csv("data/training_d1-16.csv", header=0, index_col=None)
    # print(train_set1.describe())
    print("begin to load the trainset2")
    # train_set2 = processing(trainSpan=(12,23),label=True)
    # train_set2.to_csv("data/training_d12-23.csv", header=True, index=False)
    train_set2 = pd.read_csv("data/training_d12-23.csv", header=0, index_col=None)
    # print(train_set1.describe())
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,23),label=True)
    # train_set3.to_csv("data/training_d1-23.csv", header=True, index=False)
    # print(train_set1.describe())
    print("begin to merge the trainsets")
    # train_set = pd.concat([train_set1,train_set2,train_set3],axis=0)
    train_set = pd.concat([train_set1,train_set2],axis=0)
    # train_set.to_csv("data/training_m11-23.csv", header=True, index=False)
    # train_set = pd.read_csv("data/training_m1-23.csv", header=0, index_col=None)
    # del train_set1,train_set2
    # gc.collect()
    print(train_set.describe())
    keep_feature = list(set(train_set.columns.values.tolist())-set(["user_id"]))
    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    train_label =train_set["label"]
    train_set = train_set.drop(labels=["label","user_id"], axis=1)

    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    print("begin to make prediction with plain features and without tuning parameters")
    initial_params = {
        "boosting_type":"dart",
        "n_estimators":400,
        "n_jobs":-1,
        "silent":True,
        "metric":"binary_logloss",
        'max_depth': 6,
        # "max_bin": 100,
        "num_leaves": 24,
        # 'min_child_weight': 0,
        # 'min_child_samples': 100,
        # "min_split_gain": 0.0,
        "learning_rate": 0.02,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
    }
    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    scoring = {'AUC': 'roc_auc', 'f1': "f1"}
    clf1 = GridSearchCV(LGBMClassifier(**initial_params),
                      param_grid={"n_estimators":[400,600],"num_leaves": [8,16,24],"boosting_type":["gbdt"]},
                      scoring=scoring, cv=3, refit='f1',n_jobs=-1,verbose=1)
    clf1.fit(train_set.values, train_label.values)
    # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    print(clf1.best_score_)
    print(clf1.best_params_)
    # clf1 = LGBMClassifier(**initial_params)
    # clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="auc")
    print("load the test dataset")
    test_set = processing(trainSpan=(20, 30), label=False)
    test_set.to_csv("data/testing_d20-30.csv",header=True,index=False)
    # test_set = pd.read_csv("data/testing_d15-30.csv",header=0,index_col=None)
    print("begin to make prediction")
    predict(clf1,test_set)

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = clf1.best_estimator_.feature_importances_
    print(feature_importances)
    print(feature_names)

    with open("kuaishou_stats.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["feature importance of catboost for tencent-crt", str_time])
        # writer.writerow(eval_metrics)
        feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
        for score, name in feature_score_name:
            print('{}: {}'.format(name, score))
            writer.writerow([name, score])
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    model_name = "lightgbm_" + str_time + ".pkl"
    joblib.dump(clf1, model_name)
    print("begin to tune the parameters with the selected feature")
    # paramsSpace = {
    #     "n_estimators": (200, 2000),
    #     # "boosting_type":Categorical(["dart", "rf","gbdt"]),
    #     "max_depth": (3, 8),
    #     "max_bin": (100, 300),
    #     "num_leaves": (8, 64),
    #     "min_child_weight": (1e-3, 1e3, 'log-uniform'),
    #     "min_child_samples": (10, 224),
    #     "min_split_gain": (1e-6, 1.0, 'log-uniform'),
    #     "learning_rate": (1e-6, 1.0, 'log-uniform'),
    #     "colsample_bytree": (0.6, 1.0, 'uniform'),
    #     "subsample": (0.6, 1.0, 'uniform'),
    #     'reg_alpha': (1e-3, 1e3, 'log-uniform'),
    #     'reg_lambda': (1e-3, 1e3, 'log-uniform'),
    #     "scale_pos_weight": (0.0, 1.0, 'uniform'),
    # }
    #
    # def tune_parameter(X, y, clf, params):
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     gs = BayesSearchCV(
    #         estimator=clf, search_spaces=params,
    #         scoring="f1", n_iter=60,optimizer_kwargs={"base_estimator":"GBRT"},
    #         verbose=0, n_jobs=-1, cv=3, refit=True, random_state=1234
    #     )
    #     gs.fit(X, y,callback=DeltaXStopper(0.000001))
    #     best_params = gs.best_params_
    #     best_score = gs.best_score_
    #     print(best_params)
    #     print(best_score)
    #     str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #     with open("kuaishou_stats.csv", 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["the best params for lightgbm: "])
    #         for key, value in best_params.items():
    #             writer.writerow([key, value])
    #         writer.writerow(["the best score for lightgbm: ", best_score,str_time])
    #     return gs
    #
    # model = LGBMClassifier(**initial_params)
    # clf2 = tune_parameter(train_set.values,train_label.values,model,paramsSpace)
    # print("parameter tuning over, begin to save the model!")
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #
    # model_name = "lightgbm_" + str_time + ".pkl"
    # joblib.dump(clf2, model_name)
    #
    # print("begin to process the whole dataset and ready to feed into the fitted model")
    # predict(clf2,test_set)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # print("begin to get important features")
    # feature_names = train_set.columns
    # feature_importances = clf2.best_estimator_.feature_importances_
    # print(feature_importances)
    # print(feature_names)
    #
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of catboost for tencent-crt", str_time])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
if __name__=="__main__":
    run()