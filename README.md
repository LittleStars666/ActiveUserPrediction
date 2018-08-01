### 2018中国高校计算机大数据挑战赛-快手活跃用户预测

　　第一次认真参与的机器学习比赛，复赛A榜rank15(0.91106923auc vs top1 0.91245160auc). 前前后后近两个月，在这个题目上花了不少精力，在此总结分享一下个人经验（https://github.com/hellobilllee/ActiveUserPrediction) , 希望能够给广大萌新和甚至机器学习大佬们的提供些许参考价值吧。
#### 写在前面
&emsp;&emsp;主要文件为dataprocesspy文件夹里面的create_feature_v3_parallel/nonp.py(考虑到内存与CPU的合理利用，分别写了进/线程并行和非并行版本); lgbpy文件夹里面的lgb_model.py;NN在nnpy文件夹. 运行该版本数据处理代码并行需要20G以上内存，非并行8G内存便可，生成特征数为372维，可以通过调整代码中用于生成区间特征的几个for循环内数值大小控制特征生成数量，内存消耗与特征生成数成正比。
#### 赛题背景
　　基于“快手”短视频脱敏和采样后的数据信息，预测未来一段时间活跃的用户。具体的，给定一个月的采样注册用户，以及与这些用户相关的App启动时间日志，视频拍摄时间日志和用户活动日志，预测这些用户还有哪些在下个月第一周还是活跃用户。
#### 建模思路
　　预测哪些用户活跃，哪些用户不活跃，这就是一个经典的二分类问题。但现在问题是并没有给定有标签数据，如何做预测。所以第一步，需要我们自己想方法构造标签。具体的，可以使用滑窗法。比如，使用24-30的数据来标注1-23天的数据(label train_set 1-23 with data from 24-30)，可以使用此方法构造很多训练集。测试集可以是day 1-30， 也可以是15-30，whatever. 测试集区间不能太小，可以与训练集窗口值相同，也可以不同（构造window-invariant的特征）。
#### 数据分析(dataanalysispy)
　　进入赛题，首先需要了解数据。最基本的， 每一维特征代表什么意义，每一维特征的count,min,max,mean, std, unique数目等等基本信息。 这里使用pandas 的df.describe(include="all")就可以知道这些信息了。
```python
>des_user_register= df_user_register.describe(include="all")
```
注意对于类别性特征，读取数据时需要将该特征的dtype显示设置为str，然后describe()中参数include设置为all，就可以分别得到类别型和数值型特征的统计信息了。
```python
>user_register_log = ["user_id", "register_day", "register_type", "device_type"]
>dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": str}
>df_user_register = pd.read_table("data/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
```
更深入一点，通过
```python
>print(df_user_register.groupby(by=["user_id","register_day"]).size())
```
可以得到基于user的详细信息，比如launch_day，video_create day, activity_day次数，
 使用groupby(["register_day"])得到每一天注册用户数，
```python
>print(df_user_register.groupby(by=["register_day"]).size())
```
可以在此基础之上进一步分析注册趋势，可以groupby(["register_day","register_type"])得到每一天注册类型的信息，相似的可以得到注册类型与注册设备的交互信息。通过分析数据，可以发现6，7；13，14；21，22，23，24；27，28；这几天注册用户突增，初步判定两天的为周末，21，22，23为小长假，24数据激增，进一步groupby()分析，发现这一天注册类型为3，设备类型为1，223，83的注册数据异常，将这部分数据单独提取出来
```python
>user_outliers = df_user_register["register_type"]==3)&((df_user_register["device_type"]==1)|(df_user_register["device_type"]==223)|(df_user_register["device_type"]==83))
```
然后分析后面几天这些用户是否还有出现，发现这些用户没有了任何活动，同时我将这部分用户提交到了线上进行验证过一次，评估结果出现division by zero error，说明precision和recall都为0,即这些用户都是非活跃用户，导致线上计算时出现bug，这也算是科赛平台的槽点之一吧。因为24号并不会出现在训练集，而会出现在测试集，所以如果测试集中包含此部分用户，会导致训练集与测试集分布有很大出入，所以完全可以删除此部分僵死用户，我删除后成绩提成0.001+。
#### 特征构造（dataprocesspy）
&emsp;&emsp;一般来说，数据挖掘比赛=特征构造比赛。特征构造的好坏决定了模型的上限，也基本能够保证模型的下限。虽然这个比赛，论重要性，提交数据的多少排第一，这也应该是本次比赛最大的槽点了。原始数据笼共才12维，如何从这么少的数据当中挖掘出大量信息，这需要从业务角度进行深入思考。除了基本的统计信息，如count，min,max,mean,std,gap,var, rate, ratio等，还有哪些可以挖掘的重要的信息呢。

&emsp;&emsp;我一开始尝试使用规则来做，简单的限定了最后1，2，3，4，5天的活动次数，竟然能够在A榜取得0.815成绩，最初这个成绩在稳居前100以内，而程序运行时间不过几秒钟。所以最初我觉得这个比赛应该是算法+规则取胜，就像中文分词里面，CRF，HMM， Perceptron， LSTM+CRF等等，分词算法一堆，但实际生产环境中还是能用词典就用词典．

&emsp;&emsp;所以我中期每次都是将算法得出的结果跟规则得出的结果进行合并提交，但是中期算法效果不行，所以这么尝试了一段时间后我还是将重心转到算法这块来了。后来我想了想，觉得在这个比赛里面，简单规则能够解决的问题，算法难道不能解决吗？基于树的模型不就是一堆规则吗？算法应该是能够学习到这些规则的，关键是如何将自己构造规则的思路转化为特征。这么一想之后，我一下子构建了300+特征，包括最后1，2.。。15天的launch,video_create,activity的天数、次数，以及去除窗口中倒数1，2，。。。7天后的rate和day_rate信息，还有后5，7.。。11天的gap, var，day_var信息，last_day_launch(video_create, activity)等等（具体见GitHub代码（https://github.com/hellobilllee/ActiveUserPrediction/blob/master/dataprocesspy/data_process_v9.py)）。

&emsp;&emsp;为了处理窗口大小不一致的问题，可以另外构造一套带windows权重的特征(spatial_invariant)；为了处理统一窗口中不同用户注册时间长短不一问题，可以再构造一套带register_time权重的特征(temporal_invariant)；将以上空间和时间同时考虑，可以另外构造一套temporal-spatial_invariant的特征。这套特征构造完后，基本上能够保证A榜0.819以上。一般来说，基于单个特征构造的信息我称之为一元特征，如count（rate）,var等等，基于两个以上特征构造的特征我称之为多元特征，可以通过groupby（）进行构造，我开源的代码里面有一些非常常用的、方便的、简洁的的groupby()["feature_name"].transform()构造多元特征的方法，比讨论区通过groupby().agg(),然后merge()构造多元特征方便简洁快速的多，这也是我个人结合对pandas的理解摸索出来的一些小trick。

&emsp;&emsp;一般来说，三元特征已经基本能够描述特征之间的关系了，再多往groupby()里面塞特征会极大降低程序处理速度，对于activity_log这种千万量级的数据，基本上就不要塞3个以上特征到groupby()里面了。在这个赛题里面，二元以上的特征可以在register.log中可以针对device_type和register_type构造一些，如
```python
>df_user_register_train["device_type_register_rate"] = (
>df_user_register_train.groupby(by=["device_type", "register_type"])["register_type"].transform("count")).astype(
    np.uint16)
```
这个特征描述的是每一种设备类型下的某一种注册类型有多少人注册。
还可以在activity.log中针对action_type，page， author_id和video_id之间构造一些，但是在这个文件当中构造多元特征会使得生成数据的过程慢很多倍，同时，生成的一些特征，如
```
>df_user_activity_train["user_author_action_type_num"] = (df_user_activity_train.groupby(by=["user_id", "author_id"])[
    "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
```
这个特征描述的是每一个user_id对每一个看过的视频的作者采取的不同动作个数。
但这些多元特征的重要性并不高，至少在Tree_based 和L1_based feature selection中显示，他们的重要性为0，而且删除掉这些特征线下验证对结果没有影响。所以，经过几次试验后，我决定放弃构造基于activity.log的二元特征。
构造频数（rate)特征时，最好构造相应的频率（ratio)特征，如对于上面那条rate特征，可以构造如下ratio特征：
```
>df_user_register_train["device_type_register_ratio"] = (
>df_user_register_train["device_type_register_rate"] / df_user_register_train["device_type_rate"]).astype(np.float32)
```
其中df_user_register_train["device_type_rate"]  可以通过如下代码构造：
```
>df_user_register_train["device_type_rate"] = (
>df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")).astype(np.uint16)
```
构造ratio特征可以消除windows大小不同的影响，但同时将rate留在特征集当中也是有用的。
特征构造落脚点最好是有一定的业务意义，在本赛题当中，就是要从能够完善用户画像的角度着手，我从12个原始特征当中构造的300+特征，每一个都有其实际意义，但最终成绩不理想，感觉也不能是特征背锅，因为我中期尝试了各种特征选择方式，成绩还是提升不上去。思考了一下原因，觉得主要还是影响该赛题的因素比较多，下面我就要来讲一讲影响上分的一些因素了。
#### 上分因素
影响上分的因素我总结了一下，主要有以下几个：
1. data window: 训练数据的划分。前排有的同学用1-19..23做训练，用1-30做测试，使用的是变长的windows;有的同学用1-16，8-23做训练，用15-30做预测，使用的是变长的windows。两种不同的划分方式都能取得不错的成绩，说明windows的划分是很灵活的，但有的同学使用定长的window好，有的使用变长的window好，其原因有二，一是构造的特征适应性，二是训练集的组合情况（决定了训练数据的多少以及分布）。我A榜使用的1-17..23七个窗口的训练集，1-30为测试集；B榜发现使用1-17，1-20，1-23三个窗口训练时效果会好点，同时加入a榜数据。
2. parameter tuning: 调参真是一门玄学，前期我尝试了多次使用hyperopt和BayesSearchCV()进行调参，但是效果都没有不调参好，所以后来我就基本不调参了，只稍稍调整一下LGB当中树的个数，叶子树固定为4，多了就过拟合（后期就是固定使用LGB了）。
3. model: 尝试了NN， DNN，加Gaussian噪声的NN，Catboost, LR, XGBoost, 效果都不好。 Cat boost和NN有时候效果挺好，但太容易过拟合了;XGB从个人使用的情况来看，一般干不过LGB，而且速度跟不上;LR我一般用来验证新想法是否合适，如新加的特征是否有用，新的数据划分方式是否合适，删除那些特征合适等等。一般来说，LR适合干这类事情，一是其速度快，二是其模型简单，能够获得比较靠谱的反馈，而且其结果也不错，但是对于这类成绩需要精确到小数点后七位数字的比赛来说，显然就不合适了，在实际生产环境中倒是非常不错的选择。我最后就使用了LGB，4个叶子，600，800或者1200棵树，线上成绩对树的个数比较敏感。
4. data merging: 训练集构造特别重要，从四个日志文件中划分好窗口获得特征后，如何merge需谨慎。A榜在将四个文件merge的时候需要仔细思考merge的顺序，B榜要思考是否可以A榜训练数据。考虑到register文件和launch文件所含的用户数量一致，即注册的用户都至少启动了一次，而且没有launch日志记录的缺失，所以可以先将register获得的特征merge到launch获得特征中去（其实这里merge顺序可以对调，因为两者构造特征去重后数量一致，而且用户一致），然后可以将从video_create日志中获得特征merge到之前已经merge好的数据中去，最后将activity日志merge到上一个merge好的数据当中去（activity中虽然数据量大，但是构造好特征之后去重比launch要少，因为不是所有的用户都有活动）。这样merge之后最终获得数据量即窗口中注册用户量，特征维数为从四个日志文件中所构造特征维数之和。B榜的时候，我分别尝试了只用A榜数据训练；只用B榜数据训练；使用AB榜的数据一起训练；并线上线下分别验证了效果，发现使用AB榜数据一起训练时效果最好，并且窗口划分为1-17，1-21，1-23, 测试时用B榜1-30天的数据。
5. feature engineering:构造了300+特征，不选择特征的话成绩也还行，但是一直上不去。 尝试过L1-based和Tree_based的特征选择方法，基于统计的单一变量特征选择方法也尝试过，但是效果都差不多。从业务的角度来看，我感觉每一个特征都有用，每删除一个特征我都要犹豫半天。总的来说，1000棵树的LGB选择出来分裂过的特征有一半以上，我后来就固定使用如下方式构造分类器了：
```
>clf1 = Pipeline([
    ('feature_selection', SelectFromModel(clf0,threshold=5)),
    ('classification', clf2)])
>clf1.fit(train_set.values, train_label.values)
```
　　还尝试过在构造的特征之上通过PCA，NMF，FA和FeatureAgglomeration构造一些meta-feature，就像stacking一样。加上如此构造的特征提交过一次，过拟合严重，导致直接我弃用了这种超强二次特征构造方式。其实特征的构造有时也很玄学，从实际业务意义角度着手没错，但是像hashencoding这种，构造出来的特征鬼知道有什么意义，但是放到模型当中有时候却很work,还有simhash, minhash,对于短文本型的非数值型特征进行编码，你还别说，总会有编码后的某个0，1值成为强特。
#### trick
1. 构造特征的时候指定特征数据类型（dtype)可以节约1/2到3/4的内存。pandas默认整形用int，浮点用float64，但是很多整形特征用uint8就可以了，最多不过uint32，浮点型float32就能满足要求了，甚至float16也可以使用在某些特征上，如果你清楚你所构造的特征的取值范围的话。
2. 读取过后的某些大文件后面不再使用的话及时del+gc.collect(), 可以节省一些存储空间。如果数据中间不使用但最后要使用的话，可以先保存到本地文件中，然后del+gc.collect()，最后需要使用的话再从文件中读取出来，这样处理时间会稍长点，但是对于memory 资源有限的同学还是非常有用的。
3. 使用groupby()+transform() 代替groupby().agg()+merge(）
4. 加入temporal-invariant，spatial-invariant，temporal-spatial_invariant特征
5. 统计最后或者某个日期之前活动的频率特征时可以使用apply（）结合如下函数：

```
def count_occurence(x, span):
    count_dict = Counter(list(x))
    # print(count_dict)
    occu = 0
    for i in range(span[0], span[1]):
        if i in count_dict.keys():
            occu += count_dict.get(i)
    return occu
```
span为你想要统计的某个区间。更多特征提取函数详见github（https://github.com/hellobilllee/ActiveUserPrediction/blob/master/lgbpy/lgb_v16.py):

#### 写在后面
　　通过这次比赛，学到的东西还是挺多的，主要是群里的大佬个个又有才，还乐于分享，所以整个比赛过程还是很愉快的，当然也不乏一些好演员，套路深不可测，如果没有自己的一点想法的话，真就容易被套路了。接下来打算继续冲一冲，争取进前10，拿奖金，去答辩。
