'''
    Function: meta-classifier (voting) for binary classification on balanced dataset
    @ input: pre-processed data set
    @ output: metrics and plots
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
import time
from statistics import mode
from sklearn.neural_network import MLPClassifier


def labels_mapping(label_set):
    '''
    Function: re-encode string labels into numerical labels
    :param label_set: a list of labels (with 8 different labels)
    :return: a list of labels (with only 'normal' and 'abnormal')
    '''
    label_binary = []
    for item in label_set:
        if item == 7:  # 'normal' is encoded as 7 in input data
            label_binary.append(0)  # re-encode 'normal' with 0
        else:
            label_binary.append(1)  # re-encode all malitious instances with 1
    return label_binary


def metrics_cal(label_true, label_pred):
    '''
    Function: calculate evaluation metrics
    :param label_true: the true labels
    :param label_pred: the labels predicted by classifier
    :return: return false positive rate and true positive rate
    '''
    cnf_matrix = confusion_matrix(label_true, label_pred)
    fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  # false positive
    fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)  # false negative
    tp = np.diag(cnf_matrix)  # true positive
    tn = cnf_matrix.sum() - (fp + fn + tp)  # true negative

    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    # tpr = tp / (tp + fn)  # true positive rate
    fpr = fp / (fp + tn)  # false positive rate
    recall = tp / (tp + fn)  # recall score (a.k.a. detection rate)
    # acc = (tp + tn) / (tp + tn + fp + fn)
    # fnr = fn / (fn + tp)  # false negative rate

    return fpr, recall


def metaClassifier(trainset, trainlabels, testset):
    '''
    :param trainset: training data
    :param trainlabels: labels of training data
    :param testset: testing data
    :return: labels predicted by majority (majority voting result)
    '''
    # dt_clfer = DecisionTreeClassifier().fit(trainset, trainlabels)  # train DecisionTreeClassifier()
    rf_clfer = RandomForestClassifier(max_depth=6, random_state=0).fit(trainset, trainlabels)  # RandomForestClassifier
    knn_clfer = KNeighborsClassifier(n_neighbors=11).fit(trainset, trainlabels)  # train KNN classifier
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(train_data,
                                                                                                      train_labels)

    # pred_dt = dt_clfer.predict(testset)
    pred_rf = rf_clfer.predict(testset)
    pred_knn = knn_clfer.predict(testset)
    pred_nn = nn_clf.predict(test_data)

    label_meta = []  # store labels of majority voting (for all test data)
    for i in range(len(test_data)):
        label_list = []  # temporally store labels predicted by 3 classifiers
        label_list.append(pred_nn[i])
        label_list.append(pred_rf[i])
        label_list.append(pred_knn[i])
        label_meta.append(mode(label_list))  # add majority vote to list (the most common one)

    return label_meta


start_time = time.time()

label_mapping = {'anomalous(DoSattack)': 0, 'anomalous(dataProbing)': 1, 'anomalous(malitiousControl)': 2,
                 'anomalous(malitiousOperation)': 3, 'anomalous(scan)': 4, 'anomalous(spying)': 5,
                 'anomalous(wrongSetUp)': 6, 'normal': 7}

# # dataset = pd.read_csv('dataset/encoded_unbalanced.csv', low_memory=False, header=None)  # <class 'pandas.DataFrame'>
# dataset = pd.read_csv('dataset/balanced_noTimestamp_encoded.csv', low_memory=False)  # <DataFrame'>
# dataset.iloc[0][:] = labels_mapping(dataset.iloc[0][:])


# train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)  # len(train_data)=268464
train_data = pd.read_csv('dataset/balanced_noTimestamp_mixTrain.csv')
label_index = len(train_data.iloc[0][:]) - 1

train_labels = labels_mapping(train_data.iloc[:, -1])  # separate labels of training set
train_data.drop(train_data.columns[label_index], axis=1, inplace=True)  # delete feature 'normality'

test_data = pd.read_csv('dataset/balanced_noTimestamp_mixTest.csv')  # 1993 normal, 1992 abnormal
label_index = len(train_data.iloc[0][:]) - 1

test_labels = labels_mapping(test_data.iloc[:, -1])  # separate labels of testing set
test_data.drop(test_data.columns[label_index], axis=1, inplace=True)  # delete feature 'normality'

dt_clf = DecisionTreeClassifier().fit(train_data, train_labels)  # Train DecisionTreeClassifier
predicted_test_dt = dt_clf.predict(test_data)

rf_clf = RandomForestClassifier(n_estimators=7, max_depth=4, random_state=0).fit(train_data, train_labels)
predicted_test_rf = rf_clf.predict(test_data)

knn_clf = KNeighborsClassifier(n_neighbors=3).fit(train_data, train_labels)  # Train KNN classifier
predicted_test_knn = knn_clf.predict(test_data)

# svc_clf = svm.SVC(C=0.00000001, gamma='auto', decision_function_shape='ovo').fit(train_data, train_labels)
# predicted_test_svc = svc_clf.predict(test_data)
#
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(train_data,
                                                                                                  train_labels)
predicted_test_nn = nn_clf.predict(test_data)

fpr_rf, recall_rf = metrics_cal(test_labels, predicted_test_rf)
# fpr_dt, recall_dt = metrics_cal(test_labels, predicted_test_dt)
fpr_knn, recall_knn = metrics_cal(test_labels, predicted_test_knn)
# fpr_svm, recall_svm = metrics_cal(test_labels, predicted_test_svc)
fpr_nn, recall_nn = metrics_cal(test_labels, predicted_test_nn)

meta_voting = metaClassifier(train_data, train_labels, test_data)
meta_voting_arr = np.array(meta_voting)
fpr_meta, recall_meta = metrics_cal(test_labels, meta_voting_arr)

print('Detection rate  |   False alarm rate')
# print(recall_dt, fpr_dt)
print(recall_rf, fpr_rf)
print(recall_knn, fpr_knn)
# print(recall_svm, fpr_svm)
print(recall_nn, fpr_nn)
print(recall_meta, fpr_meta)


# compare meta-classifier
plt.figure('False alarm rate for binary classes')
barWidth = 0.05
r1 = np.arange(len(fpr_rf))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
plt.bar(r1, fpr_nn, color='red', width=barWidth, label='Neural network')
plt.bar(r2, fpr_rf, color='magenta', width=barWidth, label='Random forest')
plt.bar(r3, fpr_knn, color='orange', width=barWidth, label='KNN')
plt.bar(r4, fpr_meta, color='green', width=barWidth, label='Meta-classifier')
plt.xlabel('Normality', fontweight='bold')
plt.ylabel('False alarm rate')
plt.subplots_adjust(bottom=0.1, top=0.84)
plt.xticks([r + barWidth for r in range(len(fpr_rf))], ['normal', 'abnormal'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

# plot detection rate
plt.figure('Detection rate for binary classes')
barWidth = 0.05
r1 = np.arange(len(recall_rf))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
plt.bar(r1, recall_nn, color='red', width=barWidth, label='Neural network')
plt.bar(r2, recall_rf, color='magenta', width=barWidth, label='Random forest')
plt.bar(r3, recall_knn, color='orange', width=barWidth, label='KNN')
plt.bar(r4, recall_meta, color='green', width=barWidth, label='Meta-classifier')
plt.xlabel('Normality', fontweight='bold')
plt.ylabel('Detection rate')
plt.subplots_adjust(bottom=0.1, top=0.84)
plt.xticks([r + barWidth for r in range(len(recall_rf))], ['normal', 'abnormal'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
exe_time = (time.time() - start_time)  # used time (minutes)
print('--- %d seconds ---', exe_time)

plt.show()
