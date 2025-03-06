import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def get_metrics(score, y_true):
    y_pred = np.around(score).astype(np.float32)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, score)
    pre = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    results = [acc, auc, pre, recall, f1, mcc, tnr, fnr, fpr]
    results_rounded = ["{:.3f}".format(result) for result in results]
    results_rounded_floats = [float(result_string) for result_string in results_rounded]
    return results_rounded_floats


def get_X_y():
    finance_path1 = fr"data\finance\2019.csv"
    finance_path2 = fr"data\finance\2020.csv"
    finance_path3 = fr"data\finance\2021.csv"
    finance_path4 = fr"data\finance\2022.csv"

    stock_path1 = fr"data\stock\2019"
    stock_path2 = fr"data\stock\2020"
    stock_path3 = fr"data\stock\2021"
    stock_path4 = fr"data\stock\2022"

    network_path1 = fr"data\network\network_centrality_2019.csv"
    network_path2 = fr"data\network\network_centrality_2020.csv"
    network_path3 = fr"data\network\network_centrality_2021.csv"
    network_path4 = fr"data\network\network_centrality_2022.csv"

    print("=========================>获取财务数据，网络属性数据和股票数据")

    data_dict = {}

    finance_data1 = pd.read_csv(finance_path1)
    finance_data2 = pd.read_csv(finance_path2)
    finance_data3 = pd.read_csv(finance_path3)
    finance_data4 = pd.read_csv(finance_path4)

    stock_data1, no_stock1 = process_stock(finance_data1, stock_path1, 60)

    if len(no_stock1) > 0:
        print(no_stock1, "的股票数据小于60天或不存在，需要删除")

    stock_data2, no_stock2 = process_stock(finance_data2, stock_path2, 60)
    if len(no_stock2) > 0:
        print(no_stock2, "的股票数据小于60天，需要删除")

    stock_data3, no_stock3 = process_stock(finance_data3, stock_path3, 60)
    if len(no_stock3) > 0:
        print(no_stock3, "的股票数据小于60天，需要删除")

    stock_data4, no_stock4 = process_stock(finance_data4, stock_path4, 60)
    if len(no_stock4) > 0:
        print(no_stock4, "的股票数据小于60天，需要删除")

    # 网络属性
    network_feature1 = pd.read_csv(network_path1)
    network_feature2 = pd.read_csv(network_path2)
    network_feature3 = pd.read_csv(network_path3)
    network_feature4 = pd.read_csv(network_path4)

    finance_datas = pd.concat([finance_data1, finance_data2, finance_data3, finance_data4], axis=0)
    finance_datas.reset_index(drop=True, inplace=True)
    finance_data = np.array(finance_datas.iloc[:, 4:-1])
    y = finance_datas.iloc[:, 3].astype('int')
    print(y.value_counts())
    stock_data = np.vstack((stock_data1, stock_data2, stock_data3, stock_data4))
    network_data = pd.concat([network_feature1, network_feature2, network_feature3, network_feature4])
    network_data.reset_index(drop=True, inplace=True)
    network_data = np.array(network_data.iloc[:, 1:])
    print('----------------------------------------->', finance_data.shape)
    print('----------------------------------------->', stock_data.shape)
    print('----------------------------------------->', network_data.shape)

    data_dict['finance_data'] = finance_data
    data_dict['stock_data'] = stock_data
    data_dict['network_data'] = network_data
    data_dict['y'] = y

    print("finance_shape:", finance_data.shape)
    print("network_shape:", network_data.shape)
    print("stock_shape:", stock_data.shape)
    return data_dict


def process_stock(finance_data, dir_path, min_time_stamp):
    file_name_list = os.listdir(dir_path)
    stock_code = [file_name[:9] for file_name in file_name_list]
    finance_code = list(finance_data['证券代码'].values)
    no_stock_list = []
    all_stock = []
    time_stamp = 244
    for code in finance_code:
        if code in stock_code:

            path = os.path.join(dir_path, code + '.csv')
            data = pd.read_csv(path)

            imputer = KNNImputer(n_neighbors=5)
            X1 = imputer.fit_transform(data.iloc[:, 2:])
            data = data.copy()
            data[data.columns[2:]] = X1
            company_time_stamp = data.shape[0]
            if company_time_stamp < min_time_stamp:

                no_stock_list.append(code)
            else:

                data = data.loc[:,
                       ['日开盘价', '日最高价', '日最低价', '日收盘价', '日个股交易股数', '日个股交易金额', '涨跌幅',
                        '换手率(流通股数)(%)']]
                #
                column_num = data.shape[1]
                stock_list = np.array(data).tolist()
                if company_time_stamp < time_stamp:
                    for i in range(time_stamp - company_time_stamp):
                        stock_list.append([0] * column_num)
                all_stock.append(stock_list)
        else:
            no_stock_list.append(code)
    print("-------------------------股票数据大小为{}------------------------------------".format(
        np.array(all_stock).shape))
    print("-------------------------股票数据处理完毕-----------------------------------------------")
    return np.array(all_stock), no_stock_list

# data = get_X_y()
# # print(data['finance_data'].shape)
# # print(data['network_data'].shape)
# # print(data['stock_data'].shape)
