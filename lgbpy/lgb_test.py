import csv
import datetime
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from sklearn.feature_selection import VarianceThreshold

# using this module ,one needs to deconstruct some of the features in data_process
from dataprocesspy.data_process_v7 import processing


def run(scheme_num=1,file_name="../data/data_v3/training_e"):
    train_set_ls = []
    if scheme_num ==1:
        for i in [15,16,18,19]:
            print("begin to load the dataset")
            file_name1 = file_name+"ld1-"+str(i)+".csv"
            train_set_temp = pd.read_csv(file_name1, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num ==2:
        for i in [16,19]:
            print("begin to load the dataset")
            file_name2 = file_name+"ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name2, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num ==3:
        for i in [15,16,17,19,20,21]:
            print("begin to load the dataset")
            file_name3 = file_name+ "ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name3, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    val_file_name = file_name++ "ld1-23.csv"
    val_set = pd.read_csv(val_file_name, header=0, index_col=None)
    print(val_set.describe())
    train_set = pd.concat(train_set_ls, axis=0)
    ds = train_set.describe()
    print(ds)
    keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))

    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature, inplace=True)
    val_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    print(val_set.describe())
    train_label = train_set["label"]
    val_label = val_set["label"]
    train_set = train_set.drop(labels=["label", "user_id"], axis=1)
    val_set = val_set.drop(labels=["label","user_id"], axis=1)

    # for fea in keep_feature:
    #     train_set[fea] = (train_set[fea]-train_set[fea].min())/(train_set[fea].max()-train_set[fea].min())
    #     val_set[fea] = (val_set[fea]-val_set[fea].min())/(val_set[fea].max()-val_set[fea].min())

    print("begin to make prediction with plain features and without tuning parameters")

    initial_params = {
        "colsample_bytree": 0.9,
        "learning_rate": 0.02,
        "max_bin": 200,
        # "max_depth":7,
        "min_child_samples": 64,
        "min_child_weight":0.001371129038729715,
        "min_split_gain": 0.0017264713898581718,
        "n_estimators":400,
        "num_leaves": 10,
        "reg_alpha": 100,
        "reg_lambda": 0.1,
        # "scale_pos_weight": 0.9914246775102074,
        "subsample": 0.90,
    }
    # scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LGBMClassifier(),
    #                   param_grid={"n_estimators":[200,400,600],"num_leaves": [4,5,6,8],"boosting_type":["dart"]},
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)
    # n_estimator = 400
    # num_leave = 4
    # clf1 = LGBMClassifier(n_estimators=n_estimator,num_leaves=num_leave)
    clf1 = LGBMClassifier(**initial_params)
    clf1.fit(train_set.values, train_label.values)
    print("train set report:")
    yhat = clf1.predict(train_set.values)
    print(classification_report(y_pred=yhat, y_true=train_label.values,digits=4))
    print("validation set report:")
    yhat = clf1.predict(val_set.values)
    # yhat = clf1.predict(val_set)
    print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    # feature_importances = clf1.best_estimator_.feature_importances_
    feature_importances = clf1.feature_importances_
    # print(feature_importances)
    # print(feature_names)
    feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    for score, name in feature_score_name:
        print('{}: {}'.format(name, score))
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lgb for kuaishou ", str_time])
    #     # writer.writerow(["best score", bs, "best params"])
    #     # for key, value in bp.items():
    #     #     writer.writerow([key, value])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         # print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # print("begin to tune the parameters")
    # paramsSpace = {
    #     "n_estimators": (200, 800),
    #     # "boosting_type":Categorical(["dart", "rf","gbdt"]),
    #     # "max_depth": (3, 8),
    #     "max_bin": (200, 250),
    #     "num_leaves": (3, 8),
    #     "min_child_weight": (1e-5, 1.0, 'log-uniform'),
    #     "min_child_samples": (50, 80),
    #     "min_split_gain": (1e-5, 1.0, 'log-uniform'),
    #     "learning_rate": (1e-4, 0.1, 'log-uniform'),
    #     "colsample_bytree": (0.9, 1.0, 'uniform'),
    #     "subsample": (0.8, 1.0, 'uniform'),
    #     'reg_alpha': (1.0, 1e4, 'log-uniform'),
    #     'reg_lambda': (1e-4, 1.0, 'log-uniform'),
    #     "scale_pos_weight": (0.96, 1.0, 'uniform'),
    # }
    #
    # def tune_parameter(X, y, clf, params):
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     gs = BayesSearchCV(
    #         estimator=clf, search_spaces=params,
    #         # fit_params={"eval_set":(val_set.values,val_label.values),"early_stopping_rounds":30},
    #         scoring="f1", n_iter=160, optimizer_kwargs={"base_estimator": "GBRT"},
    #         verbose=0, n_jobs=-1, cv=4, refit=True, random_state=1234
    #     )
    #     gs.fit(X, y, callback=DeltaXStopper(0.000001))
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
    #         writer.writerow(["the best score for lightgbm: ", best_score, str_time])
    #     return gs
    #
    # model = LGBMClassifier(**bp)
    # clf2 = tune_parameter(train_set.values, train_label.values, model, paramsSpace)
    # print("parameter tuning over, begin to save the model!")
    # # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # # model_name = "lightgbm_" + str_time + ".pkl"
    # # joblib.dump(clf2, model_name)
    # yhat = clf2.predict(val_set.values)
    # print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))
    # print("begin to process the whole dataset and ready to feed into the fitted model")
    # predict(clf2, test_set)
    # # predict(clf2,test_set2)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # print("begin to get important features")
    # feature_names = train_set.columns
    # feature_importances = clf2.best_estimator_.feature_importances_
    # # print(feature_importances)
    # # print(feature_names)
    #
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lgb for kuaishou", str_time])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # # print(sorted_feature_name)
    # #
    # clf1 = LGBMClassifier(**bp)
    # train_set["label"] = train_label
    # val_set["label"] = val_label
    # train_set = pd.concat([train_set, val_set], axis=0)
    # train_label = train_set["label"]
    # train_set = train_set.drop(labels=["label"], axis=1)
    # clf1.fit(train_set, train_label)
    # print("load the test dataset")
    # # # test_set = processing(trainSpan=(1, 30), label=False)
    # # # test_set.to_csv("data/testing_ld1-30.csv",header=True,index=False)
    # test_set = pd.read_csv("data/testing_eld1-30.csv",header=0,index_col=None,usecols=keep_feature+["user_id"])
    # for fea in keep_feature:
    #     test_set[fea] = (test_set[fea]-test_set[fea].min())/(test_set[fea].max()-test_set[fea].min())
    # # # test_set = processing(trainSpan=(21, 30), label=False)
    # # # test_set.to_csv("data/testing_ld21-30.csv",header=True,index=False)
    # # # test_set = pd.read_csv("data/testing_ld15-30.csv",header=0,index_col=None)
    # print("begin to make prediction")
    # predict(clf1,test_set)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # #
    # # model_name = "lightgbm_" + str_time + ".pkl"
    # # joblib.dump(clf1, model_name)
    # print("begin to tune the parameters")
    # paramsSpace = {
    #     "n_estimators": (200, 800),
    #     # "boosting_type":Categorical(["dart", "rf","gbdt"]),
    #     # "max_depth": (3, 8),
    #     "max_bin": (200, 300),
    #     "num_leaves": (3, 16),
    #     "min_child_weight": (1e-5, 1.0, 'log-uniform'),
    #     "min_child_samples": (50, 80),
    #     "min_split_gain": (1e-5, 1.0, 'log-uniform'),
    #     "learning_rate": (1e-4, 0.1, 'log-uniform'),
    #     "colsample_bytree": (0.9, 1.0, 'uniform'),
    #     "subsample": (0.8, 1.0, 'uniform'),
    #     'reg_alpha': (1.0, 1e4, 'log-uniform'),
    #     'reg_lambda': (1e-4, 1.0, 'log-uniform'),
    #     "scale_pos_weight": (0.96, 1.0, 'uniform'),
    # }
if __name__ == "__main__":
    file_name1 = "../data/data_v3/training_e"
    file_name2 = "../data/data_v4/training_e"
    run(scheme_num=1,file_name=file_name1)