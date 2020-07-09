import pandas as pd
import pyecharts as pcharts
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import create_engine
from functools import lru_cache


@lru_cache()
def get_data(city, region):
    connect_info = 'mysql+pymysql://root:123456@localhost:3306/user?charset=UTF8MB4'
    engine = create_engine(connect_info)  # use sqlalchemy to build link-engine
    sql = 'SELECT * FROM {} WHERE Region="{}"'.format(city, region)
    try:
        raw_data = pd.read_sql(sql=sql, con=engine)
        data = pd.DataFrame(raw_data)
        return data
    except:
        print("query database failed!")


def price_info(city, region):
    data = get_data(city, region)
    price_info = data['Price']
    max_price = price_info.max()
    min_price = price_info.min()
    mean_price = round(price_info.mean(), 2)
    median_price = price_info.median()
    bar = pcharts.charts.Bar()
    bar.add_xaxis(['最高房价', '最低房价', '房价均值', '房价中位数'])
    bar.add_yaxis(region, [max_price, min_price, mean_price, median_price])
    bar.set_global_opts(title_opts=pcharts.options.TitleOpts(title=region + "地区房价基本信息", subtitle='单位：万元'))
    return bar


# 房价数量分布
def price_distribute(city, region):
    data = get_data(city, region)
    price_info = data['Price']

    max_price = price_info.max()
    count = int(max_price) // 100 + 1
    sum = np.zeros(count)
    price_length = len(price_info)
    for i in range(price_length):
        temp = int(price_info[i]) // 100
        for j in range(count):
            if temp == j:
                sum[j] += 1
    xlist = []
    for i in range(len(sum)):
        xlist.append(100 * i)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(xlist)
    bar.add_yaxis(region, list(sum))
    bar.set_global_opts(xaxis_opts=pcharts.options.AxisOpts(
        axislabel_opts={"rotate": 45}),
        title_opts=pcharts.options.TitleOpts(title=region + "地区房价分布信息", subtitle='单位：万元'))
    return bar


# 区域对房价影响
def district_price(city, region):
    raw_data = get_data(city, region)
    district = raw_data.groupby('District')
    mean = district['Price'].mean()
    int_mean = float_to_int(list(mean))
    district_name = list(district.size().index)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(district_name)
    bar.add_yaxis(region, int_mean)

    bar.set_global_opts(xaxis_opts=pcharts.options.AxisOpts(
        axislabel_opts={"rotate": 45}),
        title_opts=pcharts.options.TitleOpts(title=region + "地区各区县房价信息", subtitle='单位：万元'))
    return bar


# 面积对房价影响
def size_price(city, region):
    raw_data = get_data(city, region)
    size = raw_data['Size']
    max_size = size.max()
    count = int(max_size) // 100 + 1
    sum = []
    xaxis = []
    for i in range(count * 2):
        xaxis.append((1 + i) * 50)
        temp = raw_data.loc[(i * 50 < raw_data.Size) & (raw_data.Size < (1 + i) * 50), ['Price']].mean()
        sum.append(temp[0])
    float_price = np.nan_to_num(sum)
    int_price = float_to_int(float_price)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(xaxis)
    bar.add_yaxis(region, int_price)
    bar.set_global_opts(title_opts=pcharts.options.TitleOpts(title=region + "地区房屋面积对价格影响", subtitle='单位：万元'))
    return bar


# 户型对价格的影响
def layout_price(city, region):
    raw_data = get_data(city, region)
    layout = raw_data.groupby('Layout')
    mean = layout['Price'].mean()
    int_mean = float_to_int(list(mean))
    layout_type = list(layout.size().index)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(layout_type)
    bar.add_yaxis(region, int_mean)
    bar.set_global_opts(xaxis_opts=pcharts.options.AxisOpts(
        axislabel_opts={"rotate": 45}),
        title_opts=pcharts.options.TitleOpts(title=region + "地区户型对房价影响", subtitle='单位：万元'))

    return bar


# 装修对价格影响
def renovation_price(city, region):
    raw_data = get_data(city, region)
    layout = raw_data.groupby('Renovation')
    mean = layout['Price'].mean()
    int_mean = float_to_int(list(mean))
    renovation_type = list(layout.size().index)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(renovation_type)
    bar.add_yaxis(region, int_mean)
    bar.set_global_opts(xaxis_opts=pcharts.options.AxisOpts(
        axislabel_opts={"rotate": 45}),
        title_opts=pcharts.options.TitleOpts(title=region + "地区装修对房价影响", subtitle='单位：万元'))
    return bar


# 楼层对房价的影响
def floor_price(city, region):
    raw_data = get_data(city, region)
    floor = raw_data.groupby('Floor')
    mean = floor['Price'].mean()
    int_mean = float_to_int(list(mean))
    floor_type = list(floor.size().index)

    bar = pcharts.charts.Bar()
    bar.add_xaxis(floor_type)
    bar.add_yaxis(region, int_mean)
    bar.set_global_opts(xaxis_opts=pcharts.options.AxisOpts(
        axislabel_opts={"rotate": 45, 'interval': '0'}),
        title_opts=pcharts.options.TitleOpts(title=region + "地区楼层对房价影响", subtitle='单位：万元'))
    return bar


# 划分数据集
def split_data(city, region):
    data = get_data(city, region)
    columns = ['District', 'Layout', 'Price', 'Renovation', 'Size', 'Type']
    data = pd.DataFrame(data, columns=columns)

    layout_list = list(data.groupby('Layout').size().index)
    district_list = list(data.groupby('District').size().index)
    renovation_list = list(data.groupby('Renovation').size().index)
    type_list = list(data.groupby('Type').size().index)

    layout_dic = {}
    district_dic = {}
    renovation_dic = {}
    type_dic = {}

    for i in range(len(layout_list)):
        layout_dic[layout_list[i]] = i + 1
    for i in range(len(district_list)):
        district_dic[district_list[i]] = i + 1
    for i in range(len(renovation_list)):
        renovation_dic[renovation_list[i]] = i + 1
    for i in range(len(type_list)):
        type_dic[type_list[i]] = i + 1

    # 数据集映射和清洗
    data['Layout'] = data['Layout'].map(layout_dic)
    data['District'] = data['District'].map(district_dic)
    # data['Garden']=data['Garden'].map(garden_dic)
    data['Renovation'] = data['Renovation'].map(renovation_dic)
    data['Type'] = data['Type'].map(type_dic)
    data = data.dropna(how='any')

    price = data['Price']
    price = np.array(price)
    feature = data.drop('Price', axis=1)
    feature = np.array(feature)
    feature_train, feature_test, price_train, price_test = train_test_split(feature, price, test_size=0.2,
                                                                            random_state=0)
    return [feature_train, feature_test, price_train, price_test]


# 创建模型
def model(X, y):
    # 建立模型
    cross_validator = KFold(n_splits=8, shuffle=True)
    regressor = DecisionTreeRegressor()
    params = {'max_depth': range(1, 11)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator)
    grid = grid.fit(X, y)
    return grid.best_estimator_


# 预测值
def get_predict(list):
    data = split_data(list[0], list[1])
    optimal_reg = model(data[0], data[2])
    rf = RandomForestRegressor()
    rf.fit(data[0], data[2])
    lasso = Lasso()
    lasso.fit(data[0], data[2])

    predicted_value = optimal_reg.predict(data[1])
    r21 = performance_metric(data[3], predicted_value)
    test = [[list[2], list[5], 4, list[3], list[4]]]  # 'District', 'Garden', 'Layout', 'Renovation', 'Size', 'Type'
    predict_price1 = optimal_reg.predict(test)

    predict_price2 = rf.predict(test)
    predicted_value = rf.predict(data[1])
    r22 = performance_metric(data[3], predicted_value)

    predict_price3 = lasso.predict(test)
    predicted_value = lasso.predict(data[1])
    r23 = performance_metric(data[3], predicted_value)

    temp = [int(predict_price1[0]), int(predict_price2[0]), int(predict_price3[0])]

    bar = pcharts.charts.Bar()
    bar.add_xaxis(
        ['预测1,推荐指数：' + str(round(r21, 2)), '预测2，推荐指数：' + str(round(r22, 2)), '预测3，推荐指数：' + str(round(r23, 2))])
    bar.add_yaxis('预测结果', temp)
    bar.set_global_opts(
        title_opts=pcharts.options.TitleOpts(subtitle='单位：万元'))
    return bar


def recommend(list):
    raw_data = get_data(list[0], list[1])
    data = raw_data[
        (raw_data['District'] == list[2]) & (raw_data['Size'] > list[4] - 20) & (raw_data['Size'] < list[4] + 20)]
    data = pd.DataFrame(data)
    data = data.head(5)
    price = data['Price']

    info_list = []
    for index in data.index:
        info_list.append(data.loc[index].values)
    for i in range(len(info_list)):
        info_list[i] = '小区:' + str(info_list[i][3]) + '\n户型:' + str(info_list[i][5]) + '\n价格:' + str(
            info_list[i][6]) + '\n面积:' + str(info_list[i][9]) + '平方米'

    bar = pcharts.charts.Bar()
    bar.add_xaxis(info_list)
    bar.add_yaxis("类似房源推荐", price.values.tolist())
    bar.set_global_opts(
        xaxis_opts=pcharts.options.AxisOpts(axislabel_opts={'interval': '0'}, name_textstyle_opts={'color': 'white'}),
        title_opts=pcharts.options.TitleOpts(subtitle='单位：万元'))
    return bar


# 计算R2分数
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score


def float_to_int(list):
    int_list = []
    for i in range(len(list)):
        int_list.append(int(list[i]))
    return int_list


if __name__ == '__main__':
    list = ['beijing', '东城', '广渠门', '3室2厅', 150]
    recommend(list)
    # print(sql)
