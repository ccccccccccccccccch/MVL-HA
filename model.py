import csv
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from algorithm import multiview_han
from utils import skf, get_metrics, get_X_y

random_seed = 420
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '0'  # set random seed for tensorflow-g


def main(epochs=None,
         finance_units=None,
         stock_units=None,
         network_units=None,
         attention_units=None,
         mlp_units=None,
         finance_drops=None,
         stock_drops=None,
         network_drops=None):
    data = get_X_y()

    finance_data = data["finance_data"]
    network_data = data["network_data"]
    stock_data = data["stock_data"]
    y = data['y']

    train_data_folds = []
    test_data_folds = []
    for train_index, test_index in skf.split(finance_data, y):
        finance_scalar = MinMaxScaler()
        network_scalar = MinMaxScaler()
        stock_scalar = MinMaxScaler()

        finance_scalar.fit_transform(finance_data[train_index])
        network_scalar.fit_transform(network_data[train_index])
        stock_scalar.fit_transform(stock_data[train_index].reshape(-1, stock_data[train_index].shape[-1]))

        x_train_finance_fold = finance_scalar.transform(finance_data[train_index])
        x_train_network_fold = network_scalar.transform(network_data[train_index])
        x_train_stock_fold = stock_scalar.transform(
            stock_data[train_index].reshape(-1, stock_data[train_index].shape[-1])).reshape(
            stock_data[train_index].shape[0], stock_data[train_index].shape[1], stock_data[train_index].shape[2])

        x_test_finance_fold = finance_scalar.transform(finance_data[test_index])
        x_test_network_fold = network_scalar.transform(network_data[test_index])
        x_test_stock_fold = stock_scalar.transform(
            stock_data[test_index].reshape(-1, stock_data[test_index].shape[-1])).reshape(
            stock_data[test_index].shape[0], stock_data[test_index].shape[1], stock_data[test_index].shape[2])

        train_data_folds.append([x_train_finance_fold, x_train_network_fold, x_train_stock_fold, y[train_index].values])
        test_data_folds.append([x_test_finance_fold, x_test_network_fold, x_test_stock_fold, y[test_index].values])

    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='accuracy', verbose=1, mode='max')]
    perameters = []
    all_results_train = []
    all_results_train_std = []
    all_results_test = []
    all_results_test_std = []

    if os.path.exists(f"result/multiview.csv"):
        os.remove(f"result/multiview.csv")

    # 调参
    count = 0
    class_weights = [{0: 1, 1: 1}]
    for finance in finance_units:
        for finance_drop in finance_drops:
            for stock in stock_units:
                for stock_drop in stock_drops:
                    for network in network_units:
                        for network_drop in network_drops:
                            for attention in attention_units:
                                for mlp in mlp_units:
                                    for epoch in epochs:
                                        for class_weight in class_weights:

                                            train_results = []
                                            test_results = []
                                            for i in range(5):
                                                print(
                                                    "=========================================>{}split{}".format(count,
                                                                                                                 i))
                                                model = multiview_han(finance_unit=finance, stock_unit=stock,
                                                                      network_unit=network,
                                                                      attention_unit=attention, mlp_unit=mlp,
                                                                      finance_drop=finance_drop,
                                                                      stock_drop=stock_drop, network_drop=network_drop)
                                                model.fit(x={"finance_input": train_data_folds[i][0],
                                                             "stock_input": train_data_folds[i][2],
                                                             "network_input": train_data_folds[i][1]},
                                                          y=train_data_folds[i][3],
                                                          epochs=epoch,
                                                          batch_size=32,
                                                          class_weight=class_weight,
                                                          callbacks=callbacks
                                                          )
                                                model.save(
                                                    f"saved_model/multiview_model_{count}_{i}.h5")
                                                train_score = model.predict(
                                                    {"finance_input": train_data_folds[i][0],
                                                     "stock_input": train_data_folds[i][2],
                                                     "network_input": train_data_folds[i][1]
                                                     })
                                                test_score = model.predict(
                                                    {"finance_input": test_data_folds[i][0],
                                                     "stock_input": test_data_folds[i][2],
                                                     "network_input": test_data_folds[i][1]
                                                     })
                                                train_metrics = get_metrics(train_score, train_data_folds[i][3])
                                                test_metrics = get_metrics(test_score, test_data_folds[i][3])
                                                train_results.append(train_metrics)
                                                test_results.append(test_metrics)
                                                print("第{}组参数第{}折的测试集的结果{}".format(count, i, test_metrics))
                                                print(
                                                    "第{}组参数第{}折的训练集的结果{}".format(count, i, train_metrics))

                                            train_mean_result = [round(x, 3) for x in
                                                                 np.array(train_results).mean(axis=0).tolist()]
                                            train_std_result = [round(x, 3) for x in
                                                                np.array(train_results).std(axis=0).tolist()]
                                            test_mean_result = [round(x, 3) for x in
                                                                np.array(test_results).mean(axis=0).tolist()]
                                            test_std_result = [round(x, 3) for x in
                                                               np.array(test_results).std(axis=0).tolist()]

                                            all_results_train.append(train_mean_result)
                                            all_results_train_std.append(train_std_result)
                                            all_results_test.append(test_mean_result)
                                            all_results_test_std.append(test_std_result)
                                            perameter = {"epoch": epoch, "finance_unit": finance, "stock_unit": stock,
                                                         "network_unit": network, "attention_unit": attention,
                                                         "mlp_unit": mlp, "finance_drop": finance_drop,
                                                         "stock_drop": stock_drop,
                                                         "network_drop": network_drop,
                                                         "class_weight": class_weight}
                                            perameters.append(perameter)

                                            # 保存每一组参数的结果
                                            one_perameter_results = [test_mean_result, test_std_result]
                                            with open(f"result/multiview.csv", 'a+') as csvfile:
                                                writer = csv.writer(csvfile)
                                                writer.writerow([count, finance, stock, network, attention,
                                                                 mlp, finance_drop, stock_drop, network_drop,
                                                                 epoch, class_weight])
                                                print("---------------------------------------", count)
                                                writer.writerows(one_perameter_results)
                                                writer.writerow("\n")

                                            print(
                                                "============================第{}组参数的结果============================".format(
                                                    count))
                                            print("参数为：", perameter)
                                            print("测试集五折交叉验证的结果:", test_mean_result)
                                            print("测试集的方差:", test_std_result)
                                            print("训练集五折交叉验证的结果", train_mean_result)
                                            print("训练集的方差", train_std_result)
                                            print(
                                                "=============================目前最好的结果a===============================")
                                            best_index_a = np.array(all_results_test)[:, 0].argmax()
                                            print("目前结果最好的参数是第{}组参数".format(best_index_a))
                                            print("目前最佳的参数:", perameters[best_index_a])
                                            print("目前最好的结果:", all_results_test[best_index_a])

                                            print(
                                                "=============================目前最好的结果f===============================")
                                            best_index_f = np.array(all_results_test)[:, 4].argmax()
                                            print("目前结果最好的参数是第{}组参数".format(best_index_f))
                                            print("目前最佳的参数:", perameters[best_index_f])
                                            print("目前最好的结果:", all_results_test[best_index_f])

                                            count += 1

    print("===========================调参最佳结果f================================")
    best_index_f1 = np.array(all_results_test)[:, 4].argmax()
    print("best_index:", best_index_f1)
    print("best_results:", all_results_test[best_index_f1])
    print("best_results_std", all_results_test_std[best_index_f1])
    print("best_train_results:", all_results_train[best_index_f1])
    print("best_train_results_std:", all_results_train_std[best_index_f1])
    print("best_peramter:", perameters[best_index_f1])

    # 最好的参数
    with open(f'result/best_parameters_f.txt', 'a+') as file:
        file.writelines(
            "multiview_best_parameters_ " + f"{best_index_f1}:" + str(perameters[best_index_f1]) + "\n")
    # 最好的结果
    with open(f'result/best_results_f.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow("multiview")
        writer.writerow(all_results_test[best_index_f1])
        writer.writerow(all_results_test_std[best_index_f1])

    best_index_f1 = np.array(all_results_test)[:, 4].argmax()

    best_model_paths = [f'saved_model\multiview_model_{best_index_f1}_{i}.h5' for i in range(5)]

    saved_model_dir = f'saved_model'
    all_model_files = os.listdir(saved_model_dir)

    for model_file in all_model_files:
        model_path = os.path.join(saved_model_dir, model_file)
        if model_path not in best_model_paths:
            os.remove(model_path)


if __name__ == "__main__":
    main()
