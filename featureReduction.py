import pandas as pd
import matplotlib.pyplot as plt

headers = ['sourceID', 'sourceAddress', 'sourceType', 'sourceLocation', 'destinationServiceAddress',
           'destinationServiceType', 'destinationLocation', 'accessedNodeAddress', 'accessedNodeType', 'operation',
           'value', 'timestamp', 'normality']
dataset = pd.read_csv('encoded_unbalanced.csv', header=None)

print(dataset.shape)   # 355902, 13


'''remove features which have only one value (actually no such feature)'''
delete_features = []
cols = len(dataset.iloc[0][:])
for i in range(cols):
    if len(dataset.iloc[:][i].unique()) == 1:
        del dataset[i]
        delete_features.append(i)

correlation = dataset.corr()  # 13*13
correlation.to_csv('correlation.csv', index=False, header=False)

featureSize = len(dataset.iloc[0][:])  # 40
fig, ax = plt.subplots(figsize=(featureSize, featureSize))
ax.matshow(correlation)
plt.xticks(range(len(correlation.columns)), headers, fontsize=15, rotation=270)
plt.yticks(range(len(correlation.columns)), headers, fontsize=15, rotation=0)
# plt.show()
plt.savefig('correlation_map.png')

'''remove features according to correlation between features'''
corr_size = correlation.shape[0]
delete_features_corr = []
for j in range(corr_size):
    if i != corr_size:
        for k in range(j + 1, corr_size):
            if abs(correlation.iloc[j, k]) > 0.999:
                '''
                    correlation smaller than -0.999 indicates a perfect negative correlation,
                    correlation greater than 0.999 indicates a perfect positive correlation,
                    elimate them from the feature list to avoid their influence
                '''
                del dataset[j]
                delete_features_corr.append(j)


dataset.to_csv('pre-processing_feat.csv', index=False, header=False)
pd.Series(delete_features_corr).to_csv('removed_feat_withCorr.csv', index=False)
