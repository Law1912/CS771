# regress.py by Soham Bharambe, 210264

import numpy as np

# loading the data
X_seen=np.load('X_seen.npy', allow_pickle=True, encoding='bytes')
Xtest=np.load('Xtest.npy', allow_pickle=True, encoding='bytes')
Ytest=np.load('Ytest.npy', allow_pickle=True, encoding='bytes')
class_attributes_seen=np.load('class_attributes_seen.npy', allow_pickle=True, encoding='bytes')
class_attributes_unseen=np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='bytes')

# getting the mean of X_seen
X_seen_means = [np.mean(X_seen_child, axis = 0) for X_seen_child in X_seen]
X_seen_means = np.array(X_seen_means)

# making array of labda values to be tested
lambda_values = [0.01, 0.1, 1, 10, 20, 50, 100]

# running a loop for different lambda values
for lambda_value in lambda_values:

    # making matrix W
    temp = np.dot(class_attributes_seen.T, class_attributes_seen) + lambda_value * np.identity(class_attributes_seen.shape[1])
    temp = np.linalg.inv(temp)
    temp1 = np.dot(class_attributes_seen.T, X_seen_means)
    W = np.dot(temp, temp1)

    # getting means of unseen classes by regression 
    X_unseen_means = np.dot(np.transpose(W), np.transpose(class_attributes_unseen))

    # getting predictions for Xtest
    X_unseen_means = np.transpose(X_unseen_means)
    predicted_labels = [np.argmin(np.linalg.norm(sample - X_unseen_means, ord = None, axis=1)) + 1 for sample in Xtest]
    predicted_labels = np.array(predicted_labels)

    # calculating accuracy
    accuracy = np.mean(Ytest.ravel() == predicted_labels)
    print("For lambda = ", lambda_value, "\t:\t accuracy = ", accuracy)