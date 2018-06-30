import csv
import datetime
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from data_process_v7 import processing
from sklearn.feature_selection import VarianceThreshold

def predict(clf2, test_set,param):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    # test_set = sel.transform(test_set.values)
    print("begin to make predictions")
    res = clf2.predict_proba(test_set.values)
    # res = clf2.predict_proba(test_set)
    uid["proba1"] = pd.Series(res[:, 1])
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"],inplace=True)
    uid.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "../result/uid/uid_lgb_" +param+"_"+ str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    # active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    active_users = uid["user_id"][:23727].unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "../result/sub/submission_lgb_" +param+"_"+ str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])
# using this module ,one needs to deconstruct some of the features in data_process
def run():
    use_feature = [
        "user_id","label",
        "register_day_type_rate",
        "register_day_type_ratio",
        "register_day_device_ratio",
        "register_type_ratio",
        "register_type_device",
        "register_type_device_ratio",
        "device_type_register_ratio",
        "register_day_register_type_device_ratio",

        "user_app_launch_rate",
        "user_app_launch_ratio",
        "user_app_launch_gap",
        "user_app_launch_var",

        "user_app_launch_count_b1",
        "user_app_launch_count_b2",
        "user_app_launch_count_b3",
        "user_app_launch_count_b4",
        "user_app_launch_count_b5",
        "user_app_launch_count_b6",
        "user_app_launch_count_b7",
        "user_app_launch_count_b8",
        "user_app_launch_count_b9",
        "user_app_launch_count_b10",

        # "user_app_launch_count_rb1",
        # "user_app_launch_count_rb2",
        # "user_app_launch_count_rb3",
        # "user_app_launch_count_rb4",
        # "user_app_launch_count_rb5",
        # "user_app_launch_count_rb6",
        # "user_app_launch_count_rb7",
        "user_app_launch_count_rb8",
        "user_app_launch_count_rb9",
        "user_app_launch_count_rb10",

        "user_app_launch_count_f1",
        "user_app_launch_count_f2",
        "user_app_launch_count_f3",
        "user_app_launch_count_f4",
        "user_app_launch_count_f5",
        "user_app_launch_count_f6",
        "user_app_launch_count_f7",

        "user_app_launch_count_rf1",
        "user_app_launch_count_rf2",
        "user_app_launch_count_rf3",
        # "user_app_launch_count_rf4",
        # "user_app_launch_count_rf5",
        # "user_app_launch_count_rf6",
        # "user_app_launch_count_rf7",

        "user_video_create_rate",
        "user_video_create_ratio",
        "user_video_create_day",
        "user_video_create_day_ratio",
        "user_video_create_frequency",
        "user_video_create_gap",
        "user_video_create_day_var",
        "user_video_create_var",

        "user_video_create_count_b1",
        "user_video_create_count_b2",
        "user_video_create_count_b3",
        "user_video_create_count_b4",
        "user_video_create_count_b5",
        "user_video_create_count_b6",
        "user_video_create_count_b7",

        # "user_video_create_count_rb1",
        # "user_video_create_count_rb2",
        # "user_video_create_count_rb3",
        # "user_video_create_count_rb4",
        # "user_video_create_count_rb5",
        "user_video_create_count_rb6",
        "user_video_create_count_rb7",

        "user_video_create_count_f1",
        "user_video_create_count_f2",
        "user_video_create_count_f3",
        # "user_video_create_count_f4",
        # "user_video_create_count_f5",

        "user_video_create_count_rf1",
        "user_video_create_count_rf2",
        # "user_video_create_count_rf3",
        # "user_video_create_count_rf4",
        # "user_video_create_count_rf5",

        "user_activity_rate",
        "user_activity_ratio",
        "user_activity_var",
        "user_activity_day_rate",
        "user_activity_day_ratio",
        "user_activity_frequency",
        "user_activity_gap",
        "user_activity_day_var",
        "user_page_num",
        "user_page_day_ratio",
        "user_video_num",
        "user_video_num_ratio",
        "user_author_num",
        "user_author_num_ratio",
        "user_action_type_num",

        "user_activity_count_b1",
        "user_activity_count_b2",
        "user_activity_count_b3",
        "user_activity_count_b4",
        "user_activity_count_b5",
        "user_activity_count_b6",
        "user_activity_count_b7",
        "user_activity_count_b8",
        "user_activity_count_b9",
        "user_activity_count_b10",

        # "user_activity_count_rb1",
        # "user_activity_count_rb2",
        # "user_activity_count_rb3",
        # "user_activity_count_rb4",
        # "user_activity_count_rb5",
        # "user_activity_count_rb6",
        # "user_activity_count_rb7",
        "user_activity_count_rb8",
        "user_activity_count_rb9",
        "user_activity_count_rb10",

        "user_activity_count_f1",
        "user_activity_count_f2",
        "user_activity_count_f3",
        "user_activity_count_f4",
        "user_activity_count_f5",
        "user_activity_count_f6",
        "user_activity_count_f7",

        "user_activity_count_rf1",
        "user_activity_count_rf2",
        "user_activity_count_rf3",
        # "user_activity_count_rf4",
        # "user_activity_count_rf5",
        # "user_activity_count_rf6",
        # "user_activity_count_rf7"
    ]
    # print("begin to load the trainset1")
    # train_set1 = processing(trainSpan=(1,19),label=True)
    # train_set1.to_csv("data/training_rld1-19.csv", header=True, index=False)
    # train_set1 = pd.read_csv("data/training_eld1-22.csv", header=0, index_col=None, usecols=use_feature)
    # train_set1 = pd.read_csv("data/training_rld1-19.csv", header=0, index_col=None)
    # print(train_set1.describe())
    print("begin to load the trainset2")
    # train_set2 = processing(trainSpan=(1,20),label=True)
    # train_set2.to_csv("data/training_rld1-20.csv", header=True, index=False)
    # train_set2 = pd.read_csv("data/training_rld1-20.csv", header=0, index_col=None, usecols=use_feature)
    # train_set2 = pd.read_csv("data/training_rld1-20.csv", header=0, index_col=None)
    # print(train_set2.describe())
    print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,21),label=True)
    # train_set3.to_csv("data/training_rld1-21.csv", header=True, index=False)
    # train_set3 = pd.read_csv("data/training_ld1-17.csv", header=0, index_col=None, usecols=use_feature)
    train_set3 = pd.read_csv("data/training_eld1-21_r.csv", header=0, index_col=None)
    print(train_set3.describe())
    print("begin to load the trainset4")
    # train_set4 = processing(trainSpan=(1,22),label=True)
    # train_set4.to_csv("data/training_rld1-22.csv", header=True, index=False)
    # train_set4 = pd.read_csv("data/training_rld1-19.csv", header=0, index_col=None, usecols=use_feature)
    train_set4 = pd.read_csv("data/training_eld1-22_r.csv", header=0, index_col=None)
    print(train_set4.describe())
    print("begin to load the trainset5")
    # train_set5 = processing(trainSpan=(1,21),label=True)
    # train_set5.to_csv("data/training_ld1-21.csv", header=True, index=False)
    # train_set5 = pd.read_csv("data/training_eld1-21.csv", header=0, index_col=None, usecols=use_feature)
    # print(train_set5.describe())
    print("begin to load the validation set")
    # val_set = processing(trainSpan=(1,23),label=True)
    # val_set.to_csv("data/training_rld1-23.csv", header=True, index=False)
    # val_set = pd.read_csv("data/training_eld1-23.csv", header=0, index_col=None, usecols=use_feature)
    val_set = pd.read_csv("data/training_eld1-23_r.csv", header=0, index_col=None)
    print(val_set.describe())
    train_set = pd.concat([train_set3,train_set4,val_set], axis=0)
    # train_set = pd.concat([ train_set2, train_set3, train_set4, train_set5], axis=0)
    # train_set = pd.concat([ train_set5], axis=0)
    ds = train_set.describe()
    print(ds)
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
    val_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    print(val_set.describe())
    train_label = train_set["label"]
    val_label = val_set["label"]
    train_set = train_set.drop(labels=["label", "user_id"], axis=1)
    val_set = val_set.drop(labels=["label","user_id"], axis=1)

    # keep_feature = []
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.25,random_state=42,shuffle=False)
    # train_set = train_set[keep_feature]
    # feature_names = train_set.columns

    for fea in keep_feature:
        train_set[fea] = (train_set[fea]-train_set[fea].min())/(train_set[fea].max()-train_set[fea].min())
        val_set[fea] = (val_set[fea]-val_set[fea].min())/(val_set[fea].max()-val_set[fea].min())
    # sel = VarianceThreshold(threshold=0.00001)
    # train_set_cols = train_set.columns
    # val_set_cols = train_set.columns
    # train_set = sel.fit_transform(train_set.values)
    # val_set = sel.transform(val_set.values)
    # train_x, val_set,train_label,val_label = train_test_split(train_set.values,train_label.values,test_size=0.25,random_state=42,shuffle=False)
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    print("begin to make prediction with plain features and without tuning parameters")

    # initial_params = {
    #     "colsample_bytree": 0.9310971383709499,
    #     "learning_rate": 0.01952000072830969,
    #     "max_bin": 215,
    #     # "max_depth":7,
    #     "min_child_samples": 64,
    #     "min_child_weight":0.001371129038729715,
    #     "min_split_gain": 0.0017264713898581718,
    #     "n_estimators":231,
    #     "num_leaves": 10,
    #     "reg_alpha": 673.9667614029386,
    #     "reg_lambda": 0.07753855735553884,
    #     "scale_pos_weight": 0.9914246775102074,
    #     "subsample": 0.9022953065931087,
    # }
    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    # best_f1 =0.0
    # best_params = {"n_estimators":800,"num_leaves":6}
    # for n_estimator in [400,600,800]:
    #     for num_leave in [4,6,8]:
    #         print({"n_estimators":n_estimator,"num_leaves":num_leave,"boosting_type":"dart"})
    #         clf1 = LGBMClassifier(n_estimators=n_estimator, num_leaves=num_leave, boosting_type="dart")
    #         clf1.fit(train_set.values, train_label.values)
    #         print("load the test dataset")
    #         yhat = clf1.predict(val_set.values)
    #         print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))
    #         f1 = f1_score(y_pred=yhat, y_true=val_label.values)
    #         if best_f1<f1:
    #             best_f1 = f1
    #             best_params = {"n_estimators":n_estimator,"num_leaves":num_leave,"boosting_type":"dart"}
    scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LGBMClassifier(),
    #                   param_grid={"n_estimators":[200,400,600],"num_leaves": [4,5,6,8],"boosting_type":["dart"]},
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)
    for n_estimator in [200,400,600,800]:
        for num_leave in [4,8]:
            clf1 = LGBMClassifier(n_estimators=n_estimator,num_leaves=num_leave)
            clf1.fit(train_set.values, train_label.values)
            # clf1.fit(train_set.values, train_label.values,eval_set=(val_set.values,val_label.values),early_stopping_rounds=30)
            # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
            # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
            # bs = clf1.best_score_
            # print(bs)
            # bp = clf1.best_params_
            # print(bp)

            # yhat = clf1.predict(val_set.values)
            yhat = clf1.predict(val_set.values)
            print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))

            yhat = clf1.predict(train_set.values)
            print(classification_report(y_pred=yhat, y_true=train_label.values,digits=4))
            # clf1 = LGBMClassifier(**bp)
            # train_set["label"] = train_label
            # val_set["label"] = val_label
            # train_set = pd.concat([train_set, val_set], axis=0)
            # train_set.drop_duplicates(inplace=True)
            # train_label = train_set["label"]
            # train_set = train_set.drop(labels=["label"], axis=1)
            # clf1.fit(train_set, train_label)
            print("load the test dataset")
            # test_set = processing(trainSpan=(1, 30), label=False)
            # test_set.to_csv("data/testing_eld1-30_r.csv",header=True,index=False)
            test_set = pd.read_csv("data/testing_eld1-30_r.csv",header=0,index_col=None,usecols=keep_feature+["user_id"])
            # test_set = pd.read_csv("data/testing_rld1-30.csv",header=0,index_col=None)
            for fea in keep_feature:
                test_set[fea] = (test_set[fea]-test_set[fea].min())/(test_set[fea].max()-test_set[fea].min())

            print("begin to make prediction")
            param = str(n_estimator)+"_"+str(num_leave)
            predict(clf1,test_set,param)
            #
            # # f1 = f1_score(y_pred=yhat, y_true=val_label.values)
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
            #
            with open("kuaishou_stats.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["feature importance of lgb for kuaishou ", str_time])
                # writer.writerow(["best score", bs, "best params"])
                # for key, value in bp.items():
                #     writer.writerow([key, value])
                # writer.writerow(eval_metrics)
                feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
                for score, name in feature_score_name:
                    # print('{}: {}'.format(name, score))
                    writer.writerow([name, score])
            print("begin to tune the parameters")
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
    run()