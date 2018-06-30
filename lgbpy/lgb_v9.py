import csv
import datetime
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from data_process_v6 import processing

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
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "result/uid_" + str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "result/submission_lgb_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])
# using this module ,one needs to deconstruct some of the features in data_process
def run():
    use_feature = [
        "user_id","label",
        # "register_day_rate",
        "register_type_rate",
        "register_type_device",
        "device_type_rate",
        "device_type_register",

        "user_app_launch_rate",
        # "user_app_launch_register_max_time",
        "user_app_launch_mean_time",
        # "user_app_launch_register_mean_time",
        "user_app_launch_gap",
        "user_app_launch_var",

        "user_video_create_rate",
        "user_video_create_day",
        "user_video_create_frequency",
        "user_video_create_mean_time",
        "user_video_create_gap",
        "user_video_create_var",
        # "user_video_create_register_min_time",
        # "user_video_create_register_max_time",
        # "user_video_create_register_mean_time",

        "user_activity_rate",
        "user_activity_day_rate",
        "user_activity_frequency",
        "user_activity_gap",
        "user_activity_var",
        "user_activity_mean_time",
        "user_activity_video_num",
        "user_activity_page_num",
        "user_activity_author_num",
        "user_activity_action_num",
        "user_page_num",
        "user_page_activity_num",
        "user_page_page_num",
        "user_video_num",
        "user_author_num",
        "user_author_video_num",
        "user_action_type_num",
        "user_action_type_action_type_num",
        "user_action_type_activity_num",
        # "user_activity_register_min_time",
        # "user_activity_register_max_time",
        # "user_activity_register_mean_time"
    ]
    print("begin to load the trainset1")
    train_set1 = processing(trainSpan=(1,14),label=True)
    train_set1.to_csv("data/training_ld1-14.csv", header=True, index=False)
    # train_set1 = pd.read_csv("data/training_ld1-10.csv", header=0, index_col=None, usecols=use_feature)
    print(train_set1.describe())
    print("begin to load the trainset2")
    train_set2 = processing(trainSpan=(1,17),label=True)
    train_set2.to_csv("data/training_ld1-17.csv", header=True, index=False)
    # train_set2 = pd.read_csv("data/training_ld11-20.csv", header=0, index_col=None, usecols=use_feature)
    # print(train_set2.describe())
    print("begin to load the validation set")
    train_set3 = processing(trainSpan=(1,20),label=True)
    train_set3.to_csv("data/training_ld1-20.csv", header=True, index=False)
    # val_set = pd.read_csv("data/training_ld14-23.csv", header=0, index_col=None, usecols=use_feature)
    print(train_set3.describe())
    print("begin to load the trainset4")
    train_set4 = processing(trainSpan=(1,23),label=True)
    train_set4.to_csv("data/training_ld1-23.csv", header=True, index=False)
    # train_set4 = pd.read_csv("data/training_ild1-9.csv", header=0, index_col=None)
    print("begin to merge the trainsets")
    # train_set5 = processing(trainSpan=(14,23),label=True)
    # train_set5.to_csv("data/training_ld14-23.csv", header=True, index=False)
    # train_set5 = pd.read_csv("data/training_ld14-23.csv", header=0, index_col=None)
    train_set = pd.concat([train_set1, train_set2,train_set3, train_set4], axis=0)
    # train_set = pd.concat([train_set1], axis=0)
    # train_set = pd.concat([train_set1,train_set2],axis=0)
    # train_set.to_csv("data/training_lm1-11_12-23.csv", header=True, index=False)
    # train_set = pd.read_csv("data/training_m1-23.csv", header=0, index_col=None)
    # del train_set1,train_set2
    # gc.collect()
    print(train_set.describe())
    keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))

    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature, inplace=True)
    # val_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    # print(val_set.describe())
    train_label = train_set["label"]
    # val_label =val_set["label"]
    train_set = train_set.drop(labels=["label", "user_id"], axis=1)
    # val_set = val_set.drop(labels=["label","user_id"], axis=1)
    # keep_feature = []
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.25,random_state=42,shuffle=False)
    print("begin to make prediction with plain features and without tuning parameters")
    initial_params = {
        "colsample_bytree": 0.9162118524336407,
        "learning_rate": 0.0368651327186197,
        "max_bin": 233,
        # "max_depth":7,
        "min_child_samples": 78,
        "min_child_weight":0.04510271226371328,
        "min_split_gain": 1.2202815035879434e-05,
        "n_estimators":914,
        "num_leaves": 20,
        "reg_alpha": 0.022763366314527352,
        "reg_lambda": 0.013102048688026313,
        "scale_pos_weight": 0.979101767772265,
        "subsample": 0.9826827893256738,
    }
    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    # scoring = {'AUC': 'roc_auc', 'f1': "f1"}
    scoring = "f1"
    clf1 = GridSearchCV(
        # LGBMClassifier(**initial_params),
        LGBMClassifier(),
                      param_grid={"n_estimators":[1600,800,400],"num_leaves": [4,8,12],"boosting_type":["dart"]},
                      scoring=scoring, cv=4, refit="f1",n_jobs=-1,verbose=1)
    clf1.fit(train_set.values, train_label.values)
    # clf1.fit(train_set.values, train_label.values,eval_set=(val_set.values,val_label.values),early_stopping_rounds=30)
    # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    bs = clf1.best_score_
    print(bs)
    bp = clf1.best_params_
    print(bp)
    # clf1 = LGBMClassifier(**initial_params)
    # clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="auc")
    # print("load the test dataset")
    test_set = processing(trainSpan=(1, 30), label=False)
    test_set.to_csv("data/testing_ld1-30.csv",header=True,index=False)
    # test_set = pd.read_csv("data/testing_ld15-30.csv",header=0,index_col=None)
    print("make prediction")
    # yhat = clf1.predict(val_set)
    # print(classification_report(y_pred=yhat,y_true=val_label.values))
    # test_set = pd.read_csv("data/testing_ld15-30.csv",header=0,index_col=None,usecols=use_feature)
    # test_set.drop_duplicates(inplace=True)
    print("begin to make prediction")
    predict(clf1,test_set)
    # print("load the test dataset")
    # test_set2 = processing(trainSpan=(20, 30), label=False)
    # test_set2.to_csv("data/testing_ild20-30.csv",header=True,index=False)
    # test_set2 = pd.read_csv("data/testing_ld8-30.csv",header=0,index_col=None)
    # print("begin to make prediction")
    # predict(clf1,test_set2)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = np.reshape(clf1.best_estimator_.feature_importances_,-1)
    print(feature_importances)
    print(feature_names)
    feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    for score, name in feature_score_name:
        print('{}: {}'.format(name, score))
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)
    #
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lgb for tencent-crt ", str_time])
    #     writer.writerow(["best score",clf1.best_score_,"best params"])
    #     for key, value in clf1.best_params_.items():
    #         writer.writerow([key, value])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #
    # model_name = "lightgbm_" + str_time + ".pkl"
    # joblib.dump(clf1, model_name)
    # print("begin to tune the parameters with the selected feature")
    # paramsSpace = {
    #     "n_estimators": (800, 1600),
    #     # "boosting_type":Categorical(["dart", "rf","gbdt"]),
    #     # "max_depth": (3, 8),
    #     "max_bin": (200, 300),
    #     "num_leaves": (4, 32),
    #     "min_child_weight": (1e-3, 1e3, 'log-uniform'),
    #     "min_child_samples": (50, 200),
    #     "min_split_gain": (1e-7, 1.0, 'log-uniform'),
    #     "learning_rate": (1e-7, 0.1, 'log-uniform'),
    #     "colsample_bytree": (0.9, 1.0, 'uniform'),
    #     "subsample": (0.8, 1.0, 'uniform'),
    #     'reg_alpha': (1e-3, 1e3, 'log-uniform'),
    #     'reg_lambda': (1e-4, 1.0, 'log-uniform'),
    #     "scale_pos_weight": (0.9, 1.0, 'uniform'),
    # }
    #
    # def tune_parameter(X, y, clf, params):
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     gs = BayesSearchCV(
    #         estimator=clf, search_spaces=params,
    #         # fit_params={"eval_set":(val_set.values,val_label.values),"early_stopping_rounds":30},
    #         scoring="f1", n_iter=100,optimizer_kwargs={"base_estimator":"GBRT"},
    #         verbose=0, n_jobs=-1, cv=4, refit=True, random_state=1234
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
    # model = LGBMClassifier(**bp)
    # clf2 = tune_parameter(train_set.values,train_label.values,model,paramsSpace)
    # print("parameter tuning over, begin to save the model!")
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #
    # model_name = "lightgbm_" + str_time + ".pkl"
    # joblib.dump(clf2, model_name)
    #
    # print("begin to process the whole dataset and ready to feed into the fitted model")
    # predict(clf2,test_set)
    # # predict(clf2,test_set2)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # print("begin to get important features")
    # feature_names = train_set.columns
    # feature_importances = clf2.best_estimator_.feature_importances_
    # print(feature_importances)
    # print(feature_names)
    #
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lgb for tencent-crt", str_time])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,9),label=True)
    # train_set3.to_csv("data/training_ld1-9.csv", header=True, index=False)
    # print("begin to load the trainset4")
    # train_set3 = processing(trainSpan=(1,12),label=True)
    # train_set3.to_csv("data/training_ld1-12.csv", header=True, index=False)
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(13,23),label=True)
    # train_set3.to_csv("data/training_ld13-23.csv", header=True, index=False)
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,16),label=True)
    # train_set3.to_csv("data/training_ld1-16.csv", header=True, index=False)
    # train_set3 = processing(trainSpan=(1,13),label=True)
    # train_set3.to_csv("data/training_ld1-13.csv", header=True, index=False)
    # print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(8,23),label=True)
    # train_set3.to_csv("data/training_ld8-23.csv", header=True, index=False)
    # train_set3 = processing(trainSpan=(10,23),label=True)
    # train_set3.to_csv("data/training_ld10-23.csv", header=True, index=False)
    # test_set = processing(trainSpan=(19, 30), label=False)
    # test_set.to_csv("data/testing_ld19-30.csv",header=True,index=False)
    # test_set = processing(trainSpan=(15, 30), label=False)
    # test_set.to_csv("data/testing_ld15-30.csv",header=True,index=False)
    # test_set = processing(trainSpan=(17, 30), label=False)
    # test_set.to_csv("data/testing_ld17-30.csv",header=True,index=False)
if __name__=="__main__":
    run()