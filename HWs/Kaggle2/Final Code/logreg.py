import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def save_prediction(file):
    with open(file, "r") as f:
        data = json.load(f)

    new_data = []
    for i in range(len(data)):
        if data[i] == 'neutral':
            res = 1
        elif data[i] == 'negative':
            res = 0
        elif data[i] == 'positive':
            res = 2
        new_data.append(res)
    df = pd.DataFrame(new_data, columns =['target'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'id'})
    # df.to_csv("prediction_logreg_norm1000newton7692.csv", index=False)
    df.to_csv("prediction_logreg_test.csv", index=False)


training = sparse.load_npz("Processed data/tfidf_matrix_lem_removed_words100.npz")
x_real_test = sparse.load_npz("Processed data/tfidf_matrix_lem_removed_words100_test.npz")
# print(training)
train_results = pd.read_csv("Data/train_results.csv")
train_results = list(train_results['target'])
x_train = training[:90000]
x_test = training[90000:]
y_train = train_results[:90000]
y_test = train_results[90000:]

# sc = StandardScaler(with_mean=False)
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

model = LogisticRegression(max_iter=2000, random_state = 7, penalty='l2', solver='liblinear')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_test, y_test))
# x_real_test = sc.transform(x_real_test)
pred = model.predict(x_real_test)
# with open("prediction_logreg.json", "w") as write_file:
#     json.dump(list(pred), write_file)
