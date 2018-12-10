'''
    Function:
    1. reduce the amount of normal data, keep only 7973 normal (equal to the number of anomalous records)
    2. encode the categorical features
'''

import pandas as pd
from sklearn import preprocessing


def mapping(data, column):  # encode cateogorical features
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
dataset = pd.read_csv('dataset/pre-processed-noTimestamp.csv')  # <class 'pandas.core.frame.DataFrame'>

normal_data = dataset[dataset['normality'] == 'normal']
anomalous_data = dataset[dataset['normality'] != 'normal']

frames = [normal_data.iloc[:len(anomalous_data)], anomalous_data]
# print(normal_data.iloc[:len(anomalous_data)])

balanced_dataset = pd.concat(frames)  # 15946 rows x 12 columns
balanced_dataset.to_csv('dataset/balanced_noTimestamp.csv', sep=',', header=True, index=False)

balanced_dataset = mapping(balanced_dataset, 0)  # sourceID
balanced_dataset = mapping(balanced_dataset, 1)  # sourceAddress
balanced_dataset = mapping(balanced_dataset, 2)  # sourceType
balanced_dataset = mapping(balanced_dataset, 3)  # sourceLocation
balanced_dataset = mapping(balanced_dataset, 4)  # destinationServiceAddress
balanced_dataset = mapping(balanced_dataset, 5)  # destinationServiceType
balanced_dataset = mapping(balanced_dataset, 6)  # destinationLocation
balanced_dataset = mapping(balanced_dataset, 7)  # accessedNodeAddress
balanced_dataset = mapping(balanced_dataset, 8)  # accessedNodeType
balanced_dataset = mapping(balanced_dataset, 9)  # operation
balanced_dataset = mapping(balanced_dataset, 10)  # value
balanced_dataset = mapping(balanced_dataset, 11)  # normality
# balanced_dataset.to_csv('dataset/balanced_noTimestamp_encoded.csv', sep=',', header=True, index=False)

balanced_noTimestamp_encoded = pd.read_csv('dataset/balanced_noTimestamp_encoded.csv')
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

balanced_train = pd.concat(frames_train)  # 11961 rows x 12 columns
print(len(balanced_train))
balanced_train.to_csv('dataset/balanced_noTimestamp_mixTrain.csv', sep=',', header=True, index=False)

balanced_test = pd.concat(frames_test)  # 3985 rows x 12 columns
print(len(balanced_test))
balanced_test.to_csv('dataset/balanced_noTimestamp_mixTest.csv', sep=',', header=True, index=False)

# Functioon: read the encode mapping
# labels = list(set(balanced_dataset.iloc[:, 11]))
# label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit(labels)
# balanced_dataset.iloc[:, 11] = label_encoder.transform(balanced_dataset.iloc[:, 11])
#
# le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print(le_name_mapping)
# for index in range(len(balanced_dataset)):
#     balanced_dataset = mapping(balanced_dataset, index)


