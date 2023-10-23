import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, accuracy_score, recall_score
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler


def test_NaiveBayes(x_train1, x_test, y_train1, y_test, x_real_test):
    MNB = MultinomialNB(alpha=1)
    NBClassifier = MNB.partial_fit(x_train1, y_train1, classes=['negative', 'positive', 'neutral'])
    predictions = NBClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    print(a)
    p = NBClassifier.predict(x_train1)
    aa = accuracy_score(y_train1, p)
    print(aa)

    pred = NBClassifier.predict(x_real_test)
    with open("prediction_naive.json", "w") as write_file:
        json.dump(list(pred), write_file)
    return a, p, r


training = sparse.load_npz("Processed data/bow_matrix_lem_removed_words100.npz")
x_real_test = sparse.load_npz("Processed data/bow_matrix_lem_removed_words100_test.npz")
# print(training)

train_results = pd.read_csv("Data/train_results.csv")
train_results = list(train_results['target'])
x_train1 = training[:40000]
# x_train2 = training[40000:900000]
x_test = training[40000:]
y_train1 = train_results[:40000]
# y_train2 = train_results[40000:900000]
y_test = train_results[40000:]
print(training.shape)
# print(x_train[0])
print(x_real_test.shape)
# M = sparse.csr_matrix((training.shape[0], x_real_test.shape[1]))
# M[:560175, :] += x_real_test
# print(a.shape)
# print(a)
# print(x_real_test)
# print(M[:, 5000])

# with open("Processed data/bow_words_lem_removed_words100.json") as f:
#     train_words = json.load(f)
# with open("Processed data/bow_words_lem_removed_words100_test.json") as f:
#     test_words = json.load(f)
# for word in train_words:
#     if word not in test_words:
#         training.
# sc = StandardScaler(with_mean=False)
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# x_real_test = sc.transform(x_real_test)


test_NaiveBayes(x_train1, x_test, y_train1, y_test, x_real_test)
