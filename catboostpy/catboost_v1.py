import csv
import datetime
import hyperopt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import stats
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from data_process import processing
def select_best_feature(df,y,sorted_feature_name):
    auc_ls = []
    for k in range(8,len(sorted_feature_name),1):
        selected_k_feature = sorted_feature_name[:k]
        print(selected_k_feature)
        train_len = int(len(df[selected_k_feature])*0.75)
        # train_set = df[selected_k_feature]
        # train_len = int(len(df[selected_k_feature])*0.75)
        # category_cols = [fea for fea in selected_k_feature if not fea.endswith("bin")]
        # train default classifier
        # categorical_features_indices = [df[selected_k_feature].columns.get_loc(i) for i in category_cols]
        model = CatBoostClassifier(iterations=50, random_seed=42, verbose=2).fit(X=df[selected_k_feature].iloc[:train_len],y=y[:train_len])
        metrics = model.eval_metrics(Pool(df[selected_k_feature].iloc[train_len:],y[train_len:]), ['AUC'])
        mean_auc = sum(metrics['AUC'])/float(len(metrics['AUC']))
        print((k,mean_auc))
        auc_ls.append((k,mean_auc))
    sorted_ll = sorted(auc_ls,key=lambda x: x[1],reverse=True)
    print(sorted_ll)
    best_k = sorted_ll[0][0]
    print(best_k)
    selected_k_feature = sorted_feature_name[:best_k]
    print(selected_k_feature)
    with open("tencent_stats.csv",'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["the selected best k features"])
        writer.writerow(selected_k_feature)
    return selected_k_feature
def predict(clf2, test_set):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    # if isinstance(selector,RFECV):
    #     test_set_new = selector.transform(test_set.values)
    # elif isinstance(selector,list):
    #     test_set_new = test_set[selector]
    # else:
    #     test_set_new = test_set
    print("begin to make predictions")
    res = clf2.predict(test_set.values)
    uid["y_hat"] = pd.Series(res)
    uid["label"] = uid.groupby(by=["user_id"])["y_hat"].transform(lambda x: stats.mode(x)[0])
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    uid_file = "result/uid_" + str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    active_users = (uid.loc[uid["label"] == 1]).user_id.unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])

def run():
    train_set = processing(trainSpan=(1,23),label=True)
    train_label =train_set["label"]
    train_set = train_set.drop(labels=["label","user_id"], axis=1)

    print("begin to make prediction with plain features and without tuning parameters")
    initial_params =  {
        "verbose":2,
        "loss_function":"Logloss",
        "eval_metric":"AUC",
        "iterations":200,
        "random_seed":42,
        "learning_rate":0.02,
        "one_hot_max_size":2,
        "depth":8,
        "border_count":128,
        "thread_count":4,
        # "class_weights":[0.1,1.8],
        "l2_leaf_reg":6,
        "use_best_model":True,
        # "save_snapshot":True,
        "leaf_estimation_method":'Newton',
        "od_type":'Iter',
        "od_wait":20,
        # "od_pval":0.0000001,
        # "used_ram_limit":1024*1024*1024*12,
        # "max_ctr_complexity":3,
        # "model_size_reg":10,
    }
    clf1 = CatBoostClassifier(**initial_params)
    train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y))
    # clf.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="map")
    # selector = RFECV(clf1, step=1, cv=3,scoring="f1",verbose=1,n_jobs=-1)
    # selector.fit(train_set.values, train_label.values)
    # train_set_new = selector.transform(train_set.values)
    print("load the test dataset")
    test_set = processing(trainSpan=(1, 30), label=False)
    print("begin to make prediction")
    predict(clf1, test_set)

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = clf1.feature_importances_
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

    print("begin to select important features")
    selected_k_feature = select_best_feature(train_set, train_label, sorted_feature_name)
    test_feature = ["user_id"] + selected_k_feature

    test_set_new = test_set[test_feature]
    train_set_new = train_set[selected_k_feature]

    print("begin to tune the parameters with the selected feature")
    paramsSpace = {
        'depth': hyperopt.hp.quniform("depth", 6, 12, 1),
        'learning_rate': hyperopt.hp.loguniform('learning_rate', 1e-6, 1e-1),
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 1, 10,1),
        'bagging_temperature': hyperopt.hp.uniform('bagging_temperature', 0.7, 1.0),
        'rsm': hyperopt.hp.uniform('rsm', 0.8, 1.0)
    }
    train_x, val_x,train_y,val_y = train_test_split(train_set_new.values,train_label.values,test_size=0.33,random_state=42)
    def hyperopt_objective(params):
        model = CatBoostClassifier(
            n_estimators=300,
            use_best_model=True,od_type='Iter',od_wait=20,verbose=2,eval_metric='AUC',
            depth=params['depth'],learning_rate=params["learning_rate"],
            l2_leaf_reg=params['l2_leaf_reg'],bagging_temperature=params['bagging_temperature'],
            rsm=params['rsm'])
        # model.fit(train_pool_tp, eval_set=validate_pool_tp)
        model.fit(X=train_x, y=train_y,
                eval_set=(val_x, val_y))
        y_val_hat = model.predict(train_set_new.values)
        mean_auc = roc_auc_score(train_label.values, y_val_hat)
        # metrics = model.eval_metrics(validate_pool_tf, ['AUC'])
        # mean_auc = sum(metrics['AUC'])/float(len(metrics['AUC']))
        # cv_data = cv(
        #     Pool(train_set_tf, train_label, cat_features=categorical_features_indices_tf),
        #     model.get_params()
        # )
        # mean_auc = np.max(cv_data['test-AUC-mean'])
        return 1 - mean_auc  # as hyperopt minimises
    best_params = hyperopt.fmin(
        hyperopt_objective,
        space=paramsSpace,
        algo=hyperopt.tpe.suggest,
        max_evals=60,
    )
    print(best_params)
    clf2 = CatBoostClassifier(
        verbose=2,loss_function="Logloss",
        iterations=2000,eval_metric="AUC",
        random_seed=42,use_best_model=True,
        od_type='Iter',od_wait=20,
        depth=best_params['depth'],
        learning_rate=best_params["learning_rate"],l2_leaf_reg=best_params['l2_leaf_reg'],
        bagging_temperature=best_params['bagging_temperature'],rsm=best_params['rsm'])

    clf2.fit(X=train_x, y=train_y,
              eval_set=(val_x, val_y))
    print("parameter tuning over, begin to save the model!")
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    model_name = "catboost_" + str_time + ".pkl"
    clf2.save_model(model_name)
    # joblib.dump(clf2, model_name)

    print("begin to process the whole dataset and ready to feed into the fitted model")
    predict(clf2,test_set_new)

if __name__=="__main__":
    run()