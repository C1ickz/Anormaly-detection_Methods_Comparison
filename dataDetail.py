'''
    Function: data set understanding (distribution).
'''

import pandas as pd
import matplotlib.pylab as plt

dataset = pd.read_csv('dataset/mainSimulationAccessTraces_noHeader.csv', header=None,
                      dtype='unicode')  # shape: (357953, 13)

# Label distribution of all 8 labels
x = [0, 1, 2, 3, 4, 5, 6, 7]  # encode labels with numbers to make plot nicer
labels = dataset.iloc[:, -1]
labels_count = labels.value_counts()  # count each label, <class 'pandas.core.series.Series'>
plt.bar(range(len(labels_count.index)), labels_count.values)
plt.xticks(range(len(labels_count.index)), x)
plt.xlabel('Normality')
plt.ylabel('Amount of Instance')
plt.subplots_adjust(left=0.15)
plt.savefig("labels_distribution.png")
plt.show()

# Label distribution of binary labels
normal_counter = labels_count['normal']  # count 'normal'
abnormal_counter = len(dataset) - normal_counter  # count 'abnormal'
binary_labels = {'normal': normal_counter, 'abnormal': abnormal_counter}
plt.bar(range(len(binary_labels)), list(binary_labels.values()))
plt.xticks(range(len(binary_labels)), list(binary_labels.keys()))
plt.xlabel('Normality')
plt.ylabel('Amount of Instance')
plt.subplots_adjust(left=0.15)
plt.show()
# plt.savefig("binary_labels_distribution.png")
