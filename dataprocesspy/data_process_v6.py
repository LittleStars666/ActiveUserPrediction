import pandas as pd
import numpy as np
user_register_log = ["user_id","register_day","register_type","device_type"]
app_launch_log = ["user_id","app_launch_day"]
video_create_log = ["user_id","video_create_day"]
user_activity_log = ["user_id","user_activity_day","page","video_id","author_id","action_type"]

def processing(trainSpan=(1,23),label=True):
    if label:
        assert isinstance(trainSpan,tuple),"input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0]>0 and trainSpan[0]<23 and trainSpan[1]>trainSpan[0] and trainSpan[1]<=23
    else:
        assert isinstance(trainSpan,tuple),"input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0]>0 and trainSpan[0]<30 and trainSpan[1]>trainSpan[0] and trainSpan[1]<=30
    print("get users from user register log")
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type":np.uint16}
    df_user_register = pd.read_csv("data/user_register_log.csv",header=0,index_col=None,dtype=dtype_user_register)
    df_user_register_train = df_user_register.loc[(df_user_register["register_day"]>=trainSpan[0])&(df_user_register["register_day"]<=trainSpan[1])]

    df_user_register_train["register_day_rate"] = df_user_register_train.groupby(by=["register_day"])["register_day"].transform("count")
    df_user_register_train["register_day_type_rate"] = df_user_register_train.groupby(by=["register_day","register_type"])["register_type"].transform("count")
    df_user_register_train["register_day_type_ratio"] = df_user_register_train["register_day_type_rate"]/df_user_register_train["register_day_rate"]
    df_user_register_train["register_day_device_rate"] = df_user_register_train.groupby(by=["register_day","device_type"])["device_type"].transform("count")
    df_user_register_train["register_day_device_ratio"] = df_user_register_train["register_day_device_rate"]/df_user_register_train["register_day_rate"]
    df_user_register_train["register_type_rate"] = df_user_register_train.groupby(by=["register_type"])["register_type"].transform("count")
    df_user_register_train["register_type_ratio"] = df_user_register_train["register_type_rate"]/len(df_user_register_train)
    df_user_register_train["register_type_device"] = df_user_register_train.groupby(by=["register_type"])["device_type"].transform(lambda x: x.nunique())
    df_user_register_train["register_type_device_rate"] = df_user_register_train.groupby(by=["register_type","device_type"])["device_type"].transform("count")
    df_user_register_train["register_type_device_ratio"] = df_user_register_train["register_type_device_rate"]/df_user_register_train["register_type_rate"]
    df_user_register_train["device_type_rate"] = df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")
    df_user_register_train["device_type_ratio"] = df_user_register_train["device_type_rate"]/len(df_user_register_train)
    df_user_register_train["device_type_register"] = df_user_register_train.groupby(by=["device_type"])["register_type"].transform(lambda x: x.nunique())
    df_user_register_train["device_type_register_rate"] = df_user_register_train.groupby(by=["device_type","register_type"])["register_type"].transform("count")
    df_user_register_train["device_type_register_ratio"] = df_user_register_train["device_type_register_rate"]/df_user_register_train["device_type_rate"]
    df_user_register_train["register_day_type_device_rate"] = df_user_register_train.groupby(by=["register_day","register_type","device_type"])["device_type"].transform("count")
    df_user_register_train["register_day_type_device_ratio"] = df_user_register_train["register_day_type_device_rate"]/df_user_register_train["register_day_type_rate"]
    df_user_register_train["register_day_type_device_rate"] = df_user_register_train.groupby(by=["register_day","device_type","register_type"])["register_type"].transform("count")
    df_user_register_train["register_day_type_device_ratio"] = df_user_register_train["register_day_type_device_rate"]/df_user_register_train["register_day_device_rate"]

    user_register_feature = ["user_id",
                             "register_day_rate", "register_day_type_rate","register_day_type_ratio",
                             "register_day_device_rate","register_day_device_ratio",
                             "register_type_rate","register_type_ratio","register_type_device",
                             "register_type_device_rate","register_type_device_ratio",
                             "device_type_rate","device_type_ratio", "device_type_register",
                             "device_type_register_rate","device_type_register_ratio",
                             "register_day_type_device_rate","register_day_type_device_ratio",
                             "register_day_type_device_rate","register_day_type_device_ratio"
                             ]
    df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()
    df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
    ds1 = df_user_register_train.describe()
    print(ds1)
    ds1.to_csv("kuaishou_stats2.csv",mode='a')

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {
                        "user_id": np.uint32,
                        "app_launch_day":np.uint8,
                        }
    df_app_launch = pd.read_csv("data/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch = df_app_launch.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    df_app_launch_train = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
    # print(df_app_launch_train.describe())
    df_app_launch_train["user_app_launch_rate"] = (df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform("count")).astype(np.uint8)
    df_app_launch_train["user_app_launch_ratio"] = df_app_launch_train["user_app_launch_rate"]/(df_app_launch_train["app_launch_day"].nunique())
    # df_app_launch_train["app_launch_day_user_rate"] = (df_app_launch_train.groupby(by=["app_launch_day"])[
    #     "user_id"].transform("count"))
    # df_app_launch_train["app_launch_day_user_ratio"] = df_app_launch_train["app_launch_day_user_rate"]/(df_app_launch_train["user_id"].nunique())
    df_app_launch_train["user_app_launch_mean_time"] = df_app_launch_train.groupby(by=["user_id"])[
                                                                    "app_launch_day"].transform(
        lambda x: (max(x) -min(x)) / 2)
    df_app_launch_train["user_app_launch_gap"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_app_launch_train["user_app_launch_var"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: np.var(list(set(x))))
    df_app_launch_train["user_app_launch_register_max_time"] = (df_app_launch_train.groupby(by=["user_id"])[
                                                                   "app_launch_day"].transform(lambda x: max(x)) - \
                                                               df_app_launch_train["register_day"]).astype(np.uint8)
    df_app_launch_train["user_app_launch_register_mean_time"] = df_app_launch_train.groupby(by=["user_id"])[
                                                                    "app_launch_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_app_launch_train["register_day"]


    app_launch_feature = ["user_id",
                          "user_app_launch_rate","user_app_launch_ratio",
                          # "app_launch_day_user_rate","app_launch_day_user_ratio",
                          "user_app_launch_register_max_time",
                          "user_app_launch_mean_time","user_app_launch_register_mean_time",
                          "user_app_launch_gap", "user_app_launch_var"
                          ]
    df_app_launch_train = df_app_launch_train[app_launch_feature].drop_duplicates()
    ds2 = df_app_launch_train.describe()
    print(ds2)
    ds2.to_csv("kuaishou_stats2.csv", mode='a')
    print("get users from video create")
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_csv("data/video_create_log.csv",header=0,index_col=None,dtype=dtype_video_create)
    df_video_create = df_video_create.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    df_video_create_train = df_video_create.loc[
        (df_video_create["video_create_day"] >= trainSpan[0]) & (df_video_create["video_create_day"] <= trainSpan[1])]
    df_video_create_train["user_video_create_rate"] = (df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform("count")).astype(np.uint8)
    df_video_create_train["user_video_create_day_rate"] = (df_video_create_train.groupby(by=["user_id","video_create_day"])[
        "video_create_day"].transform("count")).astype(np.uint8)
    df_video_create_train["user_video_create_day_rate_max"] = (df_video_create_train.groupby(by=["user_id"])[
        "user_video_create_day_rate"].transform("max")).astype(np.uint8)
    df_video_create_train["user_video_create_day_rate_min"] = (df_video_create_train.groupby(by=["user_id"])[
        "user_video_create_day_rate"].transform("min")).astype(np.uint8)
    df_video_create_train["user_video_create_day"] = (df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_video_create_train["user_video_create_day_ratio"] = df_video_create_train["user_video_create_day"]/(df_video_create_train["video_create_day"].nunique())
    # df_video_create_train["user_create_day_video_rate"] = (df_video_create_train.groupby(by=["video_create_day"])[
    #     "user_id"].transform(lambda x: x.nunique()))
    # df_video_create_train["user_create_day_video_ratio"] = df_video_create_train["user_create_day_video_rate"]/(df_video_create_train["user_id"].nunique())
    df_video_create_train["user_video_create_frequency"] = df_video_create_train["user_video_create_rate"] / \
                                                           df_video_create_train["user_video_create_day"]

    df_video_create_train["user_video_create_mean_time"] = df_video_create_train.groupby(by=["user_id"])[
                                                                        "video_create_day"].transform(
        lambda x: (max(x) - min(x)) / 2)
    df_video_create_train["user_video_create_gap"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_video_create_train["user_video_create_var"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: np.var(list(set(x))))
    df_video_create_train["user_video_create_register_min_time"] = (df_video_create_train.groupby(by=["user_id"])[
                                                                       "video_create_day"].transform(lambda x: min(x)) - \
                                                                   df_video_create_train["register_day"]).astype(np.uint8)
    df_video_create_train["user_video_create_register_max_time"] = (df_video_create_train.groupby(by=["user_id"])[
                                                                       "video_create_day"].transform(lambda x: max(x)) - \
                                                                   df_video_create_train["register_day"]).astype(np.uint8)
    df_video_create_train["user_video_create_register_mean_time"] = df_video_create_train.groupby(by=["user_id"])[
                                                                        "video_create_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_video_create_train["register_day"]

    # print(df_video_create_train.describe())
    video_create_feature = ["user_id",
                            # "user_create_day_video_rate","user_video_create_day_ratio",
                            # "user_video_create_day_rate",
                            "user_video_create_day_rate_max","user_video_create_day_rate_min",
                            "user_video_create_rate", "user_video_create_day", "user_video_create_frequency",
                            "user_video_create_mean_time", "user_video_create_gap", "user_video_create_var",
                            "user_video_create_register_min_time","user_video_create_register_max_time","user_video_create_register_mean_time"
                            ]
    df_video_create_train = df_video_create_train[video_create_feature].drop_duplicates()
    ds3 = df_video_create_train.describe()
    print(ds3)
    ds3.to_csv("kuaishou_stats2.csv", mode='a')

    print("get users from user activity log")
    # user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
    df_user_activity = df_user_activity.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    df_user_activity_train = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
        df_user_activity["user_activity_day"] <= trainSpan[1])]
    df_user_activity_train["user_activity_rate"] = df_user_activity_train.groupby(by=["user_id"])["user_id"].transform(
        "count")
    df_user_activity_train["user_activity_day_rate"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])["user_activity_day"].transform(
        "count")
    df_user_activity_train["user_activity_day_rate_max"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])["user_activity_day_rate"].transform(
        "max")
    df_user_activity_train["user_activity_day_rate_min"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])["user_activity_day_rate"].transform(
        "min")
    df_user_activity_train.drop_duplicates(inplace=True)
    df_user_activity_train["user_activity_day_rate"] = (df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_activity_frequency"] = df_user_activity_train["user_activity_rate"] / \
                                                        df_user_activity_train["user_activity_day_rate"]
    df_user_activity_train["user_activity_gap"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_user_activity_train["user_activity_var"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: np.var(list(set(x))))
    df_user_activity_train["user_activity_mean_time"] = df_user_activity_train.groupby(by=["user_id"])[
                                                                     "user_activity_day"].transform(
        lambda x: (max(x) - min(x)) / 2)
    df_user_activity_train["user_activity_video_num"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
                                                                     "video_id"].transform(lambda x: x.nunique())
    df_user_activity_train["user_activity_video_num_max"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
                                                                     "user_activity_video_num"].transform("max")
    df_user_activity_train["user_activity_video_num_min"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
                                                                     "user_activity_video_num"].transform("min")
    # df_user_activity_train["user_activity_page_num"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
    #                                                                  "page"].transform(lambda x: x.nunique())
    # df_user_activity_train["user_activity_author_num"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
    #                                                                  "author_id"].transform(lambda x: x.nunique())
    # df_user_activity_train["user_activity_action_num"] = df_user_activity_train.groupby(by=["user_id","user_activity_day"])[
    #                                                                  "action_type"].transform(lambda x: x.nunique())
    df_user_activity_train["user_page_num"] = (df_user_activity_train.groupby(by=["user_id"])["page"].transform(
        lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_page_page_num"] = (df_user_activity_train.groupby(by=["user_id","page"])["page"].transform("count")).astype(np.uint8)
    # df_user_activity_train["user_page_activity_num"] = (df_user_activity_train.groupby(by=["user_id","page"])["user_activity_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_page_video_num"] = (df_user_activity_train.groupby(by=["user_id","page"])["video_id"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_page_author_num"] = (df_user_activity_train.groupby(by=["user_id","page"])["author_id"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_video_num"] = (df_user_activity_train.groupby(by=["user_id"])["video_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_author_num"] = (df_user_activity_train.groupby(by=["user_id"])["author_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    # df_user_activity_train["user_author_video_num"] = (df_user_activity_train.groupby(by=["user_id","author_id"])["video_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_action_type_num"] = (df_user_activity_train.groupby(by=["user_id"])[
        "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_action_type_action_type_num"] = (df_user_activity_train.groupby(by=["user_id","action_type"])[
    #     "action_type"].transform("count")).astype(np.uint8)
    # df_user_activity_train["user_action_type_activity_num"] = (df_user_activity_train.groupby(by=["user_id","action_type"])[
    #     "user_activity_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_activity_register_min_time"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                                    "user_activity_day"].transform(lambda x: min(x)) - \
                                                                df_user_activity_train["register_day"]).astype(np.uint8)
    df_user_activity_train["user_activity_register_max_time"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                                    "user_activity_day"].transform(lambda x: max(x)) - \
                                                                df_user_activity_train["register_day"]).astype(np.uint8)
    df_user_activity_train["user_activity_register_mean_time"] = df_user_activity_train.groupby(by=["user_id"])[
                                                                     "user_activity_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_user_activity_train["register_day"]
    user_activity_feature = ["user_id",
                             "user_activity_rate", "user_activity_day_rate_max","user_activity_day_rate_min", "user_activity_frequency",
                             "user_activity_gap","user_activity_var","user_activity_mean_time",
                             "user_activity_video_num_max","user_activity_video_num_min",
                             # "user_activity_page_num","user_activity_author_num","user_activity_action_num",
                             "user_page_num",
                             # "user_page_page_num","user_page_activity_num",
                              "user_video_num", "user_author_num",
                             # "user_author_video_num",
                             "user_action_type_num",
                             # "user_action_type_action_type_num","user_action_type_activity_num",
                             "user_activity_register_min_time","user_activity_register_max_time","user_activity_register_mean_time"
                             ]
    df_user_activity_train = df_user_activity_train[user_activity_feature].drop_duplicates()
    ds4 = df_user_activity_train.describe()
    print(ds4)
    ds4.to_csv("kuaishou_stats2.csv", mode='a')
    if label:
        active_user_register = (df_user_register.loc[(df_user_register["register_day"]>trainSpan[1])&(df_user_register["register_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        active_app_launch = (df_app_launch.loc[(df_app_launch["app_launch_day"] > trainSpan[1]) & (df_app_launch["app_launch_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_video_create = (df_video_create.loc[(df_video_create["video_create_day"]>trainSpan[1])&(df_video_create["video_create_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        active_user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"] > trainSpan[1]) & (df_user_activity["user_activity_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user = list(set(active_user_register+active_app_launch+active_video_create+active_user_activity))

        df_user_register_train["label"] = 0
        df_user_register_train.loc[df_user_register_train["user_id"].isin(active_user),"label"] = 1

        df_app_launch_train["label"] = 0
        df_app_launch_train.loc[df_app_launch_train["user_id"].isin(active_user),"label"] = 1

        df_video_create_train["label"] = 0
        df_video_create_train.loc[df_video_create_train["user_id"].isin(active_user),"label"] = 1

        df_user_activity_train["label"] = 0
        df_user_activity_train.loc[df_user_activity_train["user_id"].isin(active_user),"label"] = 1

    df_launch_register = df_app_launch_train.merge(df_user_register_train,how="left").fillna(0)
    # print(df_register_launch.describe())
    df_launch_register_create = df_launch_register.merge(df_video_create_train,how="left").fillna(0)
    # print(df_register_launch_create.describe())
    # df_activity_register_launch_create = df_user_activity_train.merge(df_launch_register_create,how="left").fillna(0)
    df_launch_activity_register_create = df_launch_register_create.merge(df_user_activity_train,how="left").fillna(0)
    print("before drop the duplicates of user activity log")
    print(df_launch_activity_register_create.describe())
    keep_feature = list(set(user_register_feature + app_launch_feature + video_create_feature + user_activity_feature))
    df_launch_activity_register_create.drop_duplicates(subset=keep_feature,inplace=True)
    print("after drop the duplicates of user activity log")
    ds5 = df_launch_activity_register_create.describe()
    print(ds5)
    ds5.to_csv("kuaishou_stats2.csv", mode='a')
    return df_launch_activity_register_create

if __name__=="__main__":
    train_set = processing((1,4),label=True)
    print(train_set.info())
