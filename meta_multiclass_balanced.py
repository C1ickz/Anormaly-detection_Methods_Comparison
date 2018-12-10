'''
    Function: meta-classifier for multiclass classification
    @ single classifiers of meta-classifier are: random forest, KNN, and SVM
    @ input: pre-processed dataset
'''
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
import time
from statistics import mode
from sklearn.neural_network import MLPClassifier


def metaClassifier(trainset, trainlabels, testset):
    '''
    Function: train meta-classifier
    :param trainset: training data
    :param trainlabels: labels of training data
    :param testset: testing data
    :return: labels predicted by majority (majority voting result)
    '''
    dt_clfer = DecisionTreeClassifier().fit(trainset, trainlabels)  # train DecisionTreeClassifier()
    # rf_clfer = RandomForestClassifier(max_depth=6, random_state=0).fit(trainset, trainlabels)  # RandomForestClassifier
    knn_clfer = KNeighborsClassifier(n_neighbors=11).fit(trainset, trainlabels)  # train KNN classifier
    svc_clfer = svm.SVC(gamma='auto', decision_function_shape='ovo').fit(trainset, trainlabels)  # train SVM classifier

    pred_dt = dt_clfer.predict(testset)
    # pred_rf = rf_clfer.predict(testset)
    pred_knn = knn_clfer.predict(testset)
    pred_svc = svc_clfer.predict(testset)

    label_meta = []  # list to store labels of majority voting (for all test data)
    for i in range(len(test_data)):
        label_list = []  # temporally store labels predicted by 3 classifiers
        label_list.append(pred_svc[i])
        # label_list.append(pred_rf[i])
        label_list.append(pred_dt[i])
        label_list.append(pred_knn[i])
        label_meta.append(mode(label_list))  # add majority voting to list (the most common one)

    return label_meta


def metrics_cal(label_true, label_pred):
    '''
    Function: calculate evaluation metrics
    :param label_true: the true labels
    :param label_pred: the labels predicted by classifier
    :return: return false positive rate and recall score
    '''
    cnf_matrix = confusion_matrix(label_true, label_pred)
    fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    tp = np.diag(cnf_matrix)
    tn = cnf_matrix.sum() - (fp + fn + tp)

    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)

    # tpr = tp / (tp + fn)  # true positive rate
    fpr = fp / (fp + tn)  # false positive rate
    recall = tp / (tp + fn)  # recall score (detection rate)
    # acc = (tp + tn) / (tp + tn + fp + fn)
    # fnr = fn / (fn + tp)   # false negative rate

    return fpr, recall


start_time = time.time()

# target_names = ['anomalous(DoSattack)', 'anomalous(dataProbing)', 'anomalous(malitiousControl)',
#                 'anomalous(malitiousOperation)', 'anomalous(scan)', 'anomalous(spying)', 'anomalous(wrongSetUp)',
#                 'normal']
# dataset = pd.read_csv('dataset/encoded_unbalanced.csv', low_memory=False, header=None)  # <class 'pandas.core.frame.DataFrame'>

label_mapping = {'anomalous(DoSattack)': 0, 'anomalous(dataProbing)': 1, 'anomalous(malitiousControl)': 2,
                 'anomalous(malitiousOperation)': 3, 'anomalous(scan)': 4, 'anomalous(spying)': 5,
                 'anomalous(wrongSetUp)': 6, 'normal': 7}

train_data = pd.read_csv('dataset/balanced_noTimestamp_mixTrain.csv')
label_index = len(train_data.iloc[0][:]) - 1

train_labels = train_data.iloc[:, -1]  # separate labels of training sets
train_data.drop(train_data.columns[label_index], axis=1, inplace=True)

test_data = pd.read_csv('dataset/balanced_noTimestamp_mixTest.csv')
test_labels = test_data.iloc[:, -1]  # separate labels of testing set
test_data.drop(test_data.columns[label_index], axis=1, inplace=True)

dt_clf = DecisionTreeClassifier()  # Train DecisionTreeClassifier
selector_dt = RFE(dt_clf, None, step=1).fit(train_data, train_labels)
predicted_test_dt = selector_dt.predict(test_data)

# rf_clf = RandomForestClassifier(max_depth=6, random_state=0)  # RandomForestClassifier
# selector_rf = RFE(rf_clf, None, step=1).fit(train_data, train_labels)
# predicted_test_rf = selector_rf.predict(test_data)

knn_clf = KNeighborsClassifier(n_neighbors=5).fit(train_data, train_labels)  # Train KNN classifier
predicted_test_knn = knn_clf.predict(test_data)

# Train SVM classifier
svc_clf = svm.SVC(gamma='auto', kernel='rbf', decision_function_shape='ovo', max_iter=-1, probability=False,
                  random_state=None, shrinking=True, tol=0.001, verbose=False).fit(train_data, train_labels)
predicted_test_svc = svc_clf.predict(test_data)

# nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1).fit(train_data,
#                                                                                                  train_labels)
# predicted_test_nn = nn_clf.predict(test_data)

# fpr_rf, recall_rf = metrics_cal(test_labels, predicted_test_rf)
fpr_dt, recall_dt = metrics_cal(test_labels, predicted_test_dt)
fpr_knn, recall_knn = metrics_cal(test_labels, predicted_test_knn)
fpr_svc, recall_svc = metrics_cal(test_labels, predicted_test_svc)
# fpr_nn, recall_nn = metrics_cal(test_labels, predicted_test_nn)

meta_voting = metaClassifier(train_data, train_labels, test_data)
meta_voting_arr = np.array(meta_voting)
fpr_meta, recall_meta = metrics_cal(test_labels, meta_voting_arr)

print('Detection rate  |   False alarm rate  ')
print(recall_dt, fpr_dt)
# print(recall_rf, fpr_rf)
print(recall_knn, fpr_knn)
print(recall_svc, fpr_svc)
# print(recall_nn, fpr_nn)
print(recall_meta, fpr_meta)

# compare meta-classifier
plt.figure('False alarm rate for multiclass')
barWidth = 0.2
r1 = np.arange(len(recall_knn))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
plt.bar(r1, fpr_dt, color='magenta', width=barWidth, label='Decision tree')
# plt.bar(r1, fpr_rf, color='magenta', width=barWidth, label='Random forest')
plt.bar(r2, fpr_knn, color='orange', width=barWidth, label='KNN')
plt.bar(r3, fpr_svc, color='red', width=barWidth, label='SVM')
plt.bar(r4, fpr_meta, color='green', width=barWidth, label='Meta-classifier')
plt.xlabel('Normality', fontweight='bold')
plt.ylabel('False alarm rate')
plt.subplots_adjust(bottom=0.1, top=0.84)
plt.xticks([r + barWidth for r in range(len(fpr_knn))], ['0', '1', '2', '3', '4', '5', '6', '7'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

plt.figure('Detection rate for multiclass')
barWidth = 0.2
r1 = np.arange(len(recall_knn))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
plt.bar(r1, recall_dt, color='magenta', width=barWidth, label='Decision tree')
# plt.bar(r1, recall_rf, color='magenta', width=barWidth, label='Random forest')
plt.bar(r2, recall_knn, color='orange', width=barWidth, label='KNN')
plt.bar(r3, recall_svc, color='red', width=barWidth, label='SVM')
plt.bar(r4, recall_meta, color='green', width=barWidth, label='Meta-classifier')
plt.xlabel('Normality', fontweight='bold')
plt.ylabel('Detection rate')
plt.subplots_adjust(bottom=0.1, top=0.84)
plt.xticks([r + barWidth for r in range(len(recall_knn))], ['0', '1', '2', '3', '4', '5', '6', '7'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

exe_time = (time.time() - start_time)  # used time (minutes)
print('--- %d seconds ---', exe_time)
print('--- %d minutes ---', exe_time / 60)
plt.show()
