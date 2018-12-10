'''
    Function:
        1. remove the values with 'nan'
        2. convert categorical features to numeric using OneHotEncoder.
        3. encode labels using LabelEncoder
'''

import pandas as pd
from sklearn import preprocessing
from datetime import datetime

def mapping(data, column):
    '''
    :param data: dataset to be processed (<class 'pandas.core.frame.DataFrame'>)
    :param column: the number of the column to be transferred from string to number
    :return: encoded dataset (<class 'pandas.core.frame.DataFrame'>)
    '''
    types = list(set(data.iloc[:, column]))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(types)
    data.iloc[:, column] = label_encoder.transform(data.iloc[:, column])

    return data


headers = ['sourceID', 'sourceAddress', 'sourceType', 'sourceLocation', 'destinationServiceAddress',
           'destinationServiceType', 'destinationLocation', 'accessedNodeAddress', 'accessedNodeType', 'operation',
           'value', 'timestamp', 'normality']
dataset = pd.read_csv('dataset/mainSimulationAccessTraces_noHeader.csv',
                      names=headers)  # <class 'pandas.core.frame.DataFrame'>
dataset = dataset.dropna()  # remove the traces containing 'nan' value

dataset_noTimestamp = dataset.drop('timestamp', 1)
dataset_noTimestamp.to_csv('pre-processed-noTimestamp.csv', sep=',', header=True, index=False)


# encode the categorical features
dataset = mapping(dataset_noTimestamp, 0)  # sourceID
dataset = mapping(dataset_noTimestamp, 1)  # sourceAddress
dataset = mapping(dataset_noTimestamp, 2)  # sourceType
dataset = mapping(dataset_noTimestamp, 3)  # sourceLocation
dataset = mapping(dataset_noTimestamp, 4)  # destinationServiceAddress
dataset = mapping(dataset_noTimestamp, 5)  # destinationServiceType
dataset = mapping(dataset_noTimestamp, 6)  # destinationLocation
dataset = mapping(dataset_noTimestamp, 7)  # accessedNodeAddress
dataset = mapping(dataset_noTimestamp, 8)  # accessedNodeType
dataset = mapping(dataset_noTimestamp, 9)  # operation
dataset = mapping(dataset_noTimestamp, 10)  # value
dataset = mapping(dataset_noTimestamp, 11)  # normality


# dataset.to_csv('encoded_unbalanced.csv', sep=',', header=True, index=False)

balanced_noTimestamp_encoded = pd.read_csv('dataset/encoded_unbalanced.csv')
normal_data_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 7]
train_len_normal = round(len(normal_data_encoded) * 0.75)

anomalous_DoS_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 0]
train_len_DoS = round(len(anomalous_DoS_encoded) * 0.75)

anomalous_dataProbing_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 1]
train_len_dataProbing = round(len(anomalous_dataProbing_encoded) * 0.75)

anomalous_malControl_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 2]
train_len_malControl = round(len(anomalous_malControl_encoded) * 0.75)

anomalous_malOperation_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 3]
train_len_malOperation = round(len(anomalous_malOperation_encoded) * 0.75)

anomalous_scan_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 4]
train_len_scan = round(len(anomalous_scan_encoded) * 0.75)

anomalous_spying_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 5]
train_len_spying = round(len(anomalous_spying_encoded) * 0.75)

anomalous_wrongSetUp_encoded = balanced_noTimestamp_encoded[balanced_noTimestamp_encoded['normality'] == 6]
train_len_wrongSetUp = round(len(anomalous_wrongSetUp_encoded) * 0.75)

frames_train = [normal_data_encoded.iloc[:train_len_normal],
                anomalous_DoS_encoded.iloc[:train_len_DoS],
                anomalous_dataProbing_encoded.iloc[:train_len_dataProbing],
                anomalous_malControl_encoded.iloc[:train_len_malControl],
                anomalous_malOperation_encoded.iloc[:train_len_malOperation],
                anomalous_scan_encoded.iloc[:train_len_scan],
                anomalous_spying_encoded.iloc[:train_len_spying],
                anomalous_wrongSetUp_encoded.iloc[:train_len_wrongSetUp]]


frames_test = [normal_data_encoded.iloc[train_len_normal:],
               anomalous_DoS_encoded.iloc[train_len_DoS:],
               anomalous_dataProbing_encoded.iloc[train_len_dataProbing:],
               anomalous_malControl_encoded.iloc[train_len_malControl:],
               anomalous_malOperation_encoded.iloc[train_len_malOperation:],
               anomalous_scan_encoded.iloc[train_len_scan:],
               anomalous_spying_encoded.iloc[train_len_spying:],
               anomalous_wrongSetUp_encoded.iloc[train_len_wrongSetUp:]]

balanced_train = pd.concat(frames_train)  # 266928 rows x 12 columns
print(len(balanced_train))
balanced_train.to_csv('dataset/unbalanced_noTimestamp_mixTrain.csv', sep=',', header=True, index=False)

balanced_test = pd.concat(frames_test)  # 88974 rows x 12 columns
print(len(balanced_test))
balanced_test.to_csv('dataset/unbalanced_noTimestamp_mixTest.csv', sep=',', header=True, index=False)

# processing with timestamp
# def convert_timestamp():
#     datetime_list = []
#     for timestamp in dataset['timestamp']:
#         second = timestamp / 1000.0  # 2018-03-03 00:00:00.000000
#         converted = datetime.fromtimestamp(second).strftime('%Y-%m-%d %H:%M:%S.%f')
#         datetime_list.append(converted)
#     return pd.DataFrame(datetime_list)
