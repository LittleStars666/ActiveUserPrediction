import gc
import pandas as pd
import numpy as np

user_register_log = ["user_id", "register_day", "register_type", "device_type"]
app_launch_log = ["user_id", "app_launch_day"]
video_create_log = ["user_id", "video_create_day"]
user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
def get_ratio(ls, e):
    return ls.count(e)*1.0/ len(ls)
def processing(trainSpan=(1, 23), label=True):
    if label:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 23 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 23
    else:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 30 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 30
    training_file_name = "../data/data_v2/training_eld"+str(trainSpan[0])+"-"+str(trainSpan[1])+".csv"
    testing_file_name = "../data/data_v2/testing_eld"+str(trainSpan[0])+"-"+str(trainSpan[1])+".csv"
    print("get users from user register log")
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8,
                           "device_type": np.uint16}
    df_user_register = pd.read_csv("../data/source/user_register_log.csv", header=0, index_col=None, dtype=dtype_user_register)
    df_user_register_train = df_user_register.loc[
        (df_user_register["register_day"] >= trainSpan[0]) & (df_user_register["register_day"] <= trainSpan[1])]

    df_user_register_train["register_day_device_rate"] = (
    df_user_register_train.groupby(by=["register_day", "device_type"])["device_type"].transform("count")).astype(
        np.uint16)
    df_user_register_train["device_type_rate"] = (
    df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")).astype(np.uint16)
    df_user_register_train["device_type_ratio"] = df_user_register_train["device_type_rate"] * 1.0 / len(
        df_user_register_train)
    df_user_register_train["register_day_device_type_register_rate"] = (
    df_user_register_train.groupby(by=["register_day", "device_type", "register_type"])["register_type"].transform(
        "count")).astype(np.uint16)
    df_user_register_train["register_day_device_type_register_ratio"] = df_user_register_train[
                                                                            "register_day_device_type_register_rate"] * 1.0 / \
                                                                        df_user_register_train[
                                                                            "register_day_device_rate"]

    user_register_feature = ["user_id",
                             "register_day_device_rate",
                             "device_type_ratio",
                             "register_day_device_type_register_ratio"
                             ]
    df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
    print(df_user_register_train.describe())
    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {
        "user_id": np.uint32,
        "app_launch_day": np.uint8,
    }
    df_app_launch = pd.read_csv("../data/source/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch_train = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
    df_app_launch_train["user_app_launch_last_time"] = (
    df_app_launch_train.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: trainSpan[1] - max(x))).astype(
        np.uint8)
    app_launch_feature = [
                             "user_id",
                             "user_app_launch_last_time",
                         ]
    df_app_launch_train = df_app_launch_train[app_launch_feature].drop_duplicates()
    print(df_app_launch_train.describe())
    print("get users from video create")
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_csv("../data/source/video_create_log.csv", header=0, index_col=None, dtype=dtype_video_create)

    df_video_create_train = df_video_create.loc[
        (df_video_create["video_create_day"] >= trainSpan[0]) & (df_video_create["video_create_day"] <= trainSpan[1])]
    df_video_create_train["user_video_create_last_time"] = (
    df_video_create_train.groupby(by=["user_id"])["video_create_day"].transform(
        lambda x: trainSpan[1] - max(x))).astype(np.uint8)

    video_create_feature = ["user_id",
                            "user_video_create_last_time",
                            ]
    df_video_create_train = df_video_create_train[video_create_feature].drop_duplicates()

    print("get users from user activity log")
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_csv("../data/source/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
    df_user_activity_train = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
            df_user_activity["user_activity_day"] <= trainSpan[1])]
    print(df_user_activity_train.describe())
    del df_user_register, df_app_launch, df_video_create, df_user_activity
    gc.collect()
    user_page_name_ls = []
    for i in range(0, 5):
        page_name = "user_page_ratio" + str(i)
        user_page_name_ls.append(page_name)
        df_user_activity_train[page_name] = df_user_activity_train.groupby(by=["user_id"])["page"].transform(
            lambda x: get_ratio(list(x), i))
    user_action_type_name_ls = []
    for i in range(0, 6):
        action_type_name = "user_action_type_ratio" + str(i)
        user_action_type_name_ls.append(action_type_name)
        df_user_activity_train[action_type_name] = df_user_activity_train.groupby(by=["user_id"])[
            "action_type"].transform(lambda x: get_ratio(list(x), i))
    df_user_activity_train.drop_duplicates(inplace=True)
    df_user_activity_train["user_user_activity_last_time"] = (
    df_user_activity_train.groupby(by=["user_id"])["user_activity_day"].transform(
        lambda x: trainSpan[1] - max(x))).astype(np.uint8)
    user_activity_author = df_user_activity_train["author_id"].unique().tolist()
    user_activity_feature = [  "user_id",
                               "user_user_activity_last_time",
                                 ] \
                                 + user_page_name_ls \
                                 + user_action_type_name_ls
    df_user_activity_train = df_user_activity_train[user_activity_feature].drop_duplicates()
    df_user_activity_train["user_in_author"] = 0
    df_user_activity_train.loc[df_user_activity_train["user_id"].isin(user_activity_author), "user_in_author"] = 1
    df_user_activity_train.drop_duplicates(inplace=True)
    print(df_user_activity_train.describe())

    if label:
        df_existed = pd.read_csv(training_file_name,header=0,index_col=None)
    else:
        df_existed = pd.read_csv(testing_file_name,header=0,index_col=None)

    df_launch_register = df_existed.merge(df_app_launch_train,on=["user_id"],how="left").fillna(0)
    df_launch_register = df_launch_register.merge(df_user_register_train,on=["user_id"],how="left").fillna(0)
    # print(df_register_launch.describe())
    df_launch_register_create = df_launch_register.merge(df_video_create_train,on=["user_id"],how="left").fillna(0)
    # print(df_register_launch_create.describe())
    # df_activity_register_launch_create = df_user_activity_train.merge(df_launch_register_create,how="left").fillna(0)
    df_launch_activity_register_create = df_launch_register_create.merge(df_user_activity_train,on=["user_id"],how="left").fillna(0)
    print("before drop the duplicates of user activity log")
    print(df_launch_activity_register_create.describe())
    df_launch_activity_register_create.drop_duplicates(inplace=True)
    print("after drop the duplicates of user activity log")
    ds5 = df_launch_activity_register_create.describe()
    print(ds5)
    ds5.to_csv("kuaishou_stats2.csv", mode='a')
    return df_launch_activity_register_create

#
if __name__=="__main__":
    for i in range(15,19):
        train_set = processing((1,i),label=True)
        print(train_set.info())
        training_file_save = "../data/data_v3/training_eld1" +"-" + str(i) + "_r.csv"
        train_set.to_csv(training_file_save,header=True,index=False)
