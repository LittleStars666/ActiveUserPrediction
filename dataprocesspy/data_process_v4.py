import pandas as pd
import numpy as np

user_register_log = ["user_id", "register_day", "register_type", "device_type"]
app_launch_log = ["user_id", "app_launch_day"]
video_create_log = ["user_id", "video_create_day"]
user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]


def processing(trainSpan=(1, 23), label=True):
    if label:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 23 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 23
    else:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 30 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 30
    print("get users from user register log")
    # user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {
                            "user_id": np.uint32, "register_day": np.uint8,
                           "register_type": np.uint8,"device_type": np.uint16,
                           # "register_day_rate":np.uint16,
                           # "register_type_rate":np.uint16,"register_type_device":np.uint16,
                           # "device_type_rate":np.uint16,"device_type_register":np.uint8
                           }
    df_user_register = pd.read_csv("data/user_register_log.csv", header=0, index_col=None, dtype=dtype_user_register)
    # df_user_register = pd.read_csv("data/user_register_log_global.csv", header=0, index_col=None, dtype=dtype_user_register)
    # df_user_register.drop_duplicates(inplace=True)
    # df_user_register_train = df_user_register.loc[(df_user_register["register_day"]>=trainSpan[0])&(df_user_register["register_day"]<=trainSpan[1])]
    # these are global features
    # df_user_register["register_day_rate"] = df_user_register.groupby(by=["register_day"])["register_day"].transform(
    #     "count")
    df_user_register["register_type_rate"] = df_user_register.groupby(by=["register_type"])["register_type"].transform(
        "count")
    df_user_register["register_type_device"] = df_user_register.groupby(by=["register_type"])["device_type"].transform(
        lambda x: x.nunique())
    df_user_register["device_type_rate"] = df_user_register.groupby(by=["device_type"])["device_type"].transform(
        "count")
    df_user_register["device_type_register"] = df_user_register.groupby(by=["device_type"])["register_type"].transform(
        lambda x: x.nunique())

    user_register_feature = ["user_id",
                             # "register_day_rate",
                             "register_type_rate",
                             "register_type_device", "device_type_rate", "device_type_register"
                             ]

    df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()
    df_user_register_train = df_user_register.loc[
        (df_user_register["register_day"] >= trainSpan[0]) & (df_user_register["register_day"] <= trainSpan[1])]
    df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
    ds1 = df_user_register_train.describe()
    print(ds1)
    ds1.to_csv("kuaishou_stats2.csv", mode='a')

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {
                        "user_id": np.uint32,
                        "app_launch_day":np.uint8,
                        # "user_app_launch_rate_global":np.uint8,
                          # "user_app_launch_register_min_time_global",
                          # "user_app_launch_register_max_time_global": np.uint8,
                        # "user_app_launch_register_mean_time_global": np.float,
                        #   "user_app_launch_gap_global": np.float, "user_app_launch_var_global": np.float
                        }
    # df_app_launch = pd.read_csv("data/app_launch_log_global.csv", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch = pd.read_csv("data/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch = df_app_launch.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)

    # df_app_launch["user_app_launch_rate_global"] = df_app_launch.groupby(by=["user_id"])[
    #     "app_launch_day"].transform("count")
    # df_app_launch["user_app_launch_register_min_time_global"] = df_app_launch.groupby(by=["user_id"])[
    #                                                                 "app_launch_day"].transform(lambda x: min(x)) - \
    #                                                             df_app_launch["register_day"]
    # df_app_launch["user_app_launch_register_max_time_global"] = df_app_launch.groupby(by=["user_id"])[
    #                                                                 "app_launch_day"].transform(lambda x: max(x)) - \
    #                                                             df_app_launch["register_day"]
    # df_app_launch["user_app_launch_register_mean_time_global"] = df_app_launch.groupby(by=["user_id"])[
    #                                                                  "app_launch_day"].transform(
    #     lambda x: (max(x) + min(x)) / 2) - df_app_launch["register_day"]
    # df_app_launch["user_app_launch_gap_global"] = df_app_launch.groupby(by=["user_id"])[
    #     "app_launch_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    # df_app_launch["user_app_launch_var_global"] = df_app_launch.groupby(by=["user_id"])[
    #     "app_launch_day"].transform(lambda x: np.var(x))
    # df_app_launch.to_csv("app_launch_log_global.csv",header=True,index=False)

    # df_app_launch.drop_duplicates(inplace=True)
    df_app_launch_train = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
    # print(df_app_launch_train.describe())
    df_app_launch_train["user_app_launch_rate"] = (df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform("count")).astype(np.uint8)
    # df_app_launch_train["user_app_launch_register_min_time"] = df_app_launch_train.groupby(by=["user_id"])[
    #                                                                "app_launch_day"].transform(lambda x: min(x)) - \
    #                                                            df_app_launch_train["register_day"]
    df_app_launch_train["user_app_launch_register_max_time"] = (df_app_launch_train.groupby(by=["user_id"])[
                                                                   "app_launch_day"].transform(lambda x: max(x)) - \
                                                               df_app_launch_train["register_day"]).astype(np.uint8)
    df_app_launch_train["user_app_launch_register_mean_time"] = df_app_launch_train.groupby(by=["user_id"])[
                                                                    "app_launch_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_app_launch_train["register_day"]
    # df_app_launch_train["user_app_launch_register_gap"] = df_app_launch_train["app_launch_day"]-df_app_launch_train["app_register_day"]
    df_app_launch_train["user_app_launch_gap"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_app_launch_train["user_app_launch_var"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: np.var(x))
    app_launch_feature = ["user_id",
                          # "user_app_launch_rate_global",
                          # "user_app_launch_register_min_time_global",
                          # "user_app_launch_register_max_time_global", "user_app_launch_register_mean_time_global",
                          # "user_app_launch_gap_global", "user_app_launch_var_global",
                          "user_app_launch_rate",
                          # "user_app_launch_register_min_time",
                          "user_app_launch_register_max_time", "user_app_launch_register_mean_time",
                          "user_app_launch_gap", "user_app_launch_var"
                          ]
    df_app_launch_train = df_app_launch_train[app_launch_feature].drop_duplicates()
    ds2 = df_app_launch_train.describe()
    print(ds2)
    ds2.to_csv("kuaishou_stats2.csv", mode='a')
    print("get users from video create")
    # video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {
                            "user_id":np.uint32,
                            "video_create_day": np.uint8,
                           #  "user_video_create_rate_global": np.uint8, "user_video_create_day_global": np.uint8,
                           #  "user_video_create_frequency_global": np.float,
                           #  "user_video_create_register_min_time_global": np.uint8,
                           #  "user_video_create_register_max_time_global": np.uint8,
                           #  "user_video_create_register_mean_time_global": np.float,
                           # "user_video_create_gap_global": np.float,
                           #  "user_video_create_var_global": np.float
                        }
    df_video_create = pd.read_csv("data/video_create_log.csv", header=0, index_col=None, dtype=dtype_video_create)
    df_video_create = df_video_create.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    # df_video_create = pd.read_csv("data/video_create_log_global.csv", header=0, index_col=None, dtype=dtype_video_create)
    # df_video_create["user_video_create_rate_global"] = df_video_create.groupby(by=["user_id"])[
    #     "video_create_day"].transform("count")
    # df_video_create["user_video_create_day_global"] = df_video_create.groupby(by=["user_id"])[
    #     "video_create_day"].transform(lambda x: x.nunique())
    # df_video_create["user_video_create_frequency_global"] = df_video_create["user_video_create_rate_global"] / \
    #                                                         df_video_create["user_video_create_day_global"]
    #
    # df_video_create["user_video_create_register_min_time_global"] = df_video_create.groupby(by=["user_id"])[
    #                                                                     "video_create_day"].transform(
    #     lambda x: min(x)) - \
    #                                                                 df_video_create["register_day"]
    # df_video_create["user_video_create_register_max_time_global"] = df_video_create.groupby(by=["user_id"])[
    #                                                                     "video_create_day"].transform(
    #     lambda x: max(x)) - \
    #                                                                 df_video_create["register_day"]
    # df_video_create["user_video_create_register_mean_time_global"] = df_video_create.groupby(by=["user_id"])[
    #                                                                      "video_create_day"].transform(
    #     lambda x: (max(x) + min(x)) / 2) - df_video_create["register_day"]
    # # df_video_create["user_video_create_register_mean_time"] = df_video_create["video_create_day"]-df_video_create["register_day"]
    # df_video_create["user_video_create_gap_global"] = df_video_create.groupby(by=["user_id"])[
    #     "video_create_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    # df_video_create["user_video_create_var_global"] = df_video_create.groupby(by=["user_id"])[
    #     "video_create_day"].transform(lambda x: np.var(x))

    # df_video_create.drop_duplicates(inplace=True)
    df_video_create_train = df_video_create.loc[
        (df_video_create["video_create_day"] >= trainSpan[0]) & (df_video_create["video_create_day"] <= trainSpan[1])]

    df_video_create_train["user_video_create_rate"] = (df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform("count")).astype(np.uint8)
    df_video_create_train["user_video_create_day"] = (df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_video_create_train["user_video_create_frequency"] = df_video_create_train["user_video_create_rate"] / \
                                                           df_video_create_train["user_video_create_day"]

    df_video_create_train["user_video_create_register_min_time"] = (df_video_create_train.groupby(by=["user_id"])[
                                                                       "video_create_day"].transform(lambda x: min(x)) - \
                                                                   df_video_create_train["register_day"]).astype(np.uint8)
    df_video_create_train["user_video_create_register_max_time"] = (df_video_create_train.groupby(by=["user_id"])[
                                                                       "video_create_day"].transform(lambda x: max(x)) - \
                                                                   df_video_create_train["register_day"]).astype(np.uint8)
    df_video_create_train["user_video_create_register_mean_time"] = df_video_create_train.groupby(by=["user_id"])[
                                                                        "video_create_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_video_create_train["register_day"]
    # df_video_create_train["user_video_create_register_mean_time"] = df_video_create_train["video_create_day"]-df_video_create_train["register_day"]
    df_video_create_train["user_video_create_gap"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_video_create_train["user_video_create_var"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: np.var(x))
    # print(df_video_create_train.describe())
    video_create_feature = ["user_id",
                            # "user_video_create_rate_global", "user_video_create_day_global",
                            # "user_video_create_frequency_global",
                            # "user_video_create_register_min_time_global", "user_video_create_register_max_time_global",
                            # "user_video_create_register_mean_time_global", "user_video_create_gap_global",
                            # "user_video_create_var_global",
                            "user_video_create_rate", "user_video_create_day", "user_video_create_frequency",
                            "user_video_create_register_min_time", "user_video_create_register_max_time",
                            "user_video_create_register_mean_time", "user_video_create_gap", "user_video_create_var",
                            ]
    df_video_create_train = df_video_create_train[video_create_feature].drop_duplicates()
    ds3 = df_video_create_train.describe()
    print(ds3)
    ds3.to_csv("kuaishou_stats2.csv", mode='a')

    print("get users from user activity log")
    # user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    dtype_user_activity = {
                            "user_id": np.uint32,
                              "user_activity_day": np.uint8,"page": np.uint8,
                              "video_id": np.uint32, "author_id": np.uint32, "action_type": np.uint8,
                             # "user_activity_rate_global": np.uint16, "user_activity_day_rate_global": np.uint8,
                             # "user_activity_frequency_global": np.float,
                             # "user_activity_gap_global": np.float,
                             # "user_activity_var_global": np.float,
                             # "user_activity_register_min_time_global": np.uint8,"user_activity_register_max_time_global": np.uint8,
                             # "user_activity_register_mean_time_global": np.float,
                             # "user_page_num_global": np.uint8,"user_video_num_global": np.uint16, "user_author_num_global": np.uint16,
                             # "user_action_type_num_global": np.uint8,
                             # "user_author_video_num_global": np.uint16,
                             # "user_video_action_type_num_global": np.uint8,
                             # "user_author_action_type_num_global": np.uint8,
                             # "user_page_action_type_num_global": np.uint8,
                             # "page_rate_global": np.uint32, "page_video_global": np.uint32,"page_author_global": np.uint32,
                             # "video_rate_global": np.uint32, "video_user_global": np.uint16, "video_action_type_global": np.uint8,
                             # "author_rate_global": np.uint32, "author_user_global": np.uint16, "author_video_global": np.uint16,
                             # "author_action_type_global": np.uint8,
                             # "action_type_rate_global": np.uint8
                           }
    df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
    df_user_activity = df_user_activity.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    # df_user_activity = pd.read_csv("data/user_activity_log_global.csv", header=0, index_col=None, dtype=dtype_user_activity)
    # df_user_activity = df_user_activity.sample(n=50000)
    # print(df_user_activity.describe())
    # df_user_activity.drop_duplicates(inplace=True)
    # print(df_user_activity.describe())
    # df_user_activity["user_activity_rate_global"] = df_user_activity.groupby(by=["user_id"])["user_id"].transform(
    #     "count")
    # df_user_activity["user_activity_day_rate_global"] = df_user_activity.groupby(by=["user_id"])[
    #     "user_activity_day"].transform(lambda x: x.nunique())
    # df_user_activity["user_activity_frequency_global"] = df_user_activity["user_activity_rate_global"] / df_user_activity[
    #     "user_activity_day_rate_global"]
    # df_user_activity["user_activity_gap_global"] = df_user_activity.groupby(by=["user_id"])[
    #     "user_activity_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    # df_user_activity["user_activity_var_global"] = df_user_activity.groupby(by=["user_id"])[
    #     "user_activity_day"].transform(lambda x: np.var(x))
    # df_user_activity["user_activity_register_min_time_global"] = df_user_activity.groupby(by=["user_id"])[
    #                                                                  "user_activity_day"].transform(lambda x: min(x)) - \
    #                                                              df_user_activity["register_day"]
    # df_user_activity["user_activity_register_max_time_global"] = df_user_activity.groupby(by=["user_id"])[
    #                                                                  "user_activity_day"].transform(lambda x: max(x)) - \
    #                                                              df_user_activity["register_day"]
    # df_user_activity["user_activity_register_mean_time_global"] = df_user_activity.groupby(by=["user_id"])[
    #                                                                   "user_activity_day"].transform(
    #     lambda x: (max(x) + min(x)) / 2) - df_user_activity["register_day"]
    #
    # df_user_activity["user_page_num_global"] = df_user_activity.groupby(by=["user_id"])["page"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["user_video_num_global"] = df_user_activity.groupby(by=["user_id"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["user_author_num_global"] = df_user_activity.groupby(by=["user_id"])["author_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["user_action_type_num_global"] = df_user_activity.groupby(by=["user_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["user_author_video_num_global"] = df_user_activity.groupby(by=["user_id", "author_id"])[
    #     "video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["user_video_action_type_num_global"] = df_user_activity.groupby(by=["user_id", "video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["user_author_action_type_num_global"] = df_user_activity.groupby(by=["user_id", "author_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["user_page_action_type_num_global"] = df_user_activity.groupby(by=["user_id", "page"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["page_rate_global"] = df_user_activity.groupby(by=["page"])["page"].transform("count")
    # df_user_activity["page_video_global"] = df_user_activity.groupby(by=["page"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["page_author_global"] = df_user_activity.groupby(by=["page"])["author_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["video_rate_global"] = df_user_activity.groupby(by=["video_id"])["video_id"].transform(
    #     "count")
    # df_user_activity["video_user_global"] = df_user_activity.groupby(by=["video_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["video_action_type_global"] = df_user_activity.groupby(by=["video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["author_rate_global"] = df_user_activity.groupby(by=["video_id"])["author_id"].transform(
    #     "count")
    # df_user_activity["author_user_global"] = df_user_activity.groupby(by=["author_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["author_video_global"] = df_user_activity.groupby(by=["author_id"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity["author_action_type_global"] = df_user_activity.groupby(by=["author_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity["action_type_rate_global"] = df_user_activity.groupby(by=["action_type"])[
    #     "action_type"].transform("count")

    df_user_activity_train = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
                df_user_activity["user_activity_day"] <= trainSpan[1])]

    df_user_activity_train["user_activity_rate"] = (df_user_activity_train.groupby(by=["user_id"])["user_id"].transform(
        "count")).astype(np.uint16)
    print("before drop the duplicates of user activity log")
    print(df_user_activity_train.describe())
    df_user_activity_train.drop_duplicates(inplace=True)
    print("after drop the duplicates of user activity log")
    print(df_user_activity_train.describe())
    df_user_activity_train["user_activity_day_rate"] = (df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_activity_frequency"] = df_user_activity_train["user_activity_rate"] / \
                                                        df_user_activity_train["user_activity_day_rate"]
    df_user_activity_train["user_activity_gap"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_user_activity_train["user_activity_var"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: np.var(x))
    df_user_activity_train["user_activity_register_min_time"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                                    "user_activity_day"].transform(lambda x: min(x)) - \
                                                                df_user_activity_train["register_day"]).astype(np.uint8)
    df_user_activity_train["user_activity_register_max_time"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                                    "user_activity_day"].transform(lambda x: max(x)) - \
                                                                df_user_activity_train["register_day"]).astype(np.uint8)
    df_user_activity_train["user_activity_register_mean_time"] = df_user_activity_train.groupby(by=["user_id"])[
                                                                     "user_activity_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_user_activity_train["register_day"]

    df_user_activity_train["user_page_num"] = (df_user_activity_train.groupby(by=["user_id"])["page"].transform(
        lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_video_num"] = (df_user_activity_train.groupby(by=["user_id"])["video_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_author_num"] = (df_user_activity_train.groupby(by=["user_id"])["author_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_action_type_num"] = (df_user_activity_train.groupby(by=["user_id"])[
        "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_author_video_num"] = (df_user_activity_train.groupby(by=["user_id", "author_id"])[
    #     "video_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    # df_user_activity_train["user_video_action_type_num"] = (df_user_activity_train.groupby(by=["user_id", "video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity_train["user_author_action_type_num"] = (df_user_activity_train.groupby(by=["user_id", "author_id"])[
        "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["user_page_action_type_num"] = (df_user_activity_train.groupby(by=["user_id", "page"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity_train["page_rate"] = df_user_activity_train.groupby(by=["page"])["page"].transform("count")
    # df_user_activity_train["page_video"] = df_user_activity_train.groupby(by=["page"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["page_author"] = df_user_activity_train.groupby(by=["page"])["author_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["video_rate"] = df_user_activity_train.groupby(by=["video_id"])["video_id"].transform(
    #     "count")
    # df_user_activity_train["video_user"] = df_user_activity_train.groupby(by=["video_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["video_action_type"] = df_user_activity_train.groupby(by=["video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity_train["author_rate"] = df_user_activity_train.groupby(by=["video_id"])["author_id"].transform(
    #     "count")
    # df_user_activity_train["author_user"] = df_user_activity_train.groupby(by=["author_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["author_video"] = df_user_activity_train.groupby(by=["author_id"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["author_action_type"] = df_user_activity_train.groupby(by=["author_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity_train["action_type_rate"] = df_user_activity_train.groupby(by=["action_type"])[
    #     "action_type"].transform("count")

    user_activity_feature = ["user_id",
                             # "user_activity_rate_global", "user_activity_day_rate_global",
                             # "user_activity_frequency_global",
                             # "user_activity_gap_global","user_activity_var_global",
                             # "user_activity_register_min_time_global", "user_activity_register_max_time_global",
                             # "user_activity_register_mean_time_global",
                             # "user_page_num_global", "user_video_num_global", "user_author_num_global",
                             # "user_action_type_num_global",
                             # "user_author_video_num_global", "user_video_action_type_num_global",
                             # "user_author_action_type_num_global",
                             # "user_page_action_type_num_global",
                             # "page_rate_global", "page_video_global", "page_author_global",
                             # "video_rate_global", "video_user_global", "video_action_type_global",
                             # "author_rate_global", "author_user_global", "author_video_global",
                             # "author_action_type_global",
                             # "action_type_rate_global",

                             "user_activity_rate", "user_activity_day_rate", "user_activity_frequency",
                             "user_activity_gap","user_activity_var",
                             "user_activity_register_min_time", "user_activity_register_max_time",
                             "user_activity_register_mean_time",
                             "user_page_num", "user_video_num", "user_author_num", "user_action_type_num",
                             # "user_author_video_num", "user_video_action_type_num", "user_author_action_type_num",
                             # "user_page_action_type_num",
                             ]
    df_user_activity_train = df_user_activity_train[user_activity_feature].drop_duplicates()
    ds4 = df_user_activity_train.describe()
    print(ds4)
    ds4.to_csv("kuaishou_stats2.csv", mode='a')
    if label:
        active_user_register = (df_user_register.loc[(df_user_register["register_day"] > trainSpan[1]) & (
                    df_user_register["register_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_app_launch = (df_app_launch.loc[(df_app_launch["app_launch_day"] > trainSpan[1]) & (
                    df_app_launch["app_launch_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_video_create = (df_video_create.loc[(df_video_create["video_create_day"] > trainSpan[1]) & (
                    df_video_create["video_create_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"] > trainSpan[1]) & (
                    df_user_activity["user_activity_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user = list(set(active_user_register + active_app_launch + active_video_create + active_user_activity))

        df_user_register_train["label"] = 0
        df_user_register_train.loc[df_user_register_train["user_id"].isin(active_user), "label"] = 1

        df_app_launch_train["label"] = 0
        df_app_launch_train.loc[df_app_launch_train["user_id"].isin(active_user), "label"] = 1

        df_video_create_train["label"] = 0
        df_video_create_train.loc[df_video_create_train["user_id"].isin(active_user), "label"] = 1

        df_user_activity_train["label"] = 0
        df_user_activity_train.loc[df_user_activity_train["user_id"].isin(active_user), "label"] = 1

    # df_register_launch = df_user_register_train.merge(df_app_launch_train,how="left")
    df_launch_register = df_app_launch_train.merge(df_user_register_train, how="left").fillna(0)
    # print(df_register_launch.describe())
    df_launch_register_create = df_launch_register.merge(df_video_create_train, how="left").fillna(0)
    # print(df_register_launch_create.describe())
    df_activity_register_launch_create = df_user_activity_train.merge(df_launch_register_create, how="left").fillna(0)
    # df_launch_register_create_activity = df_launch_register_create.merge(df_user_activity_train, how="left").fillna(0)
    print("before drop the duplicates of the merged active dataset")
    # print(df_activity_register_launch_create.describe())
    keep_feature = list(set(user_register_feature + app_launch_feature + video_create_feature + user_activity_feature))
    df_activity_register_launch_create.drop_duplicates(subset=keep_feature, inplace=True)
    print("after drop the duplicates of the merged active dataset")
    ds5 = df_activity_register_launch_create.describe()
    print(ds5)
    ds5.to_csv("kuaishou_stats2.csv", mode='a')
    return df_activity_register_launch_create

# if __name__=="__main__":
#     train_set = processing((1,5),label=True)
#     print(train_set.info())
