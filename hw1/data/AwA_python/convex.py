# convex.py by Soham Bharambe, 210264

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

# getting similarities in unseen and seen class using class attributes
similarities = np.dot(class_attributes_unseen, np.transpose(class_attributes_seen))

# normalizing similarities
similarities_sum = np.sum(similarities, axis=1)
similarities_normalized = similarities / similarities_sum.reshape(-1, 1)

# getting unseen class means using the above calculated arguments
X_unseen_means = np.dot(similarities_normalized, X_seen_means)

# getting the prediction labels for the Xtest using prototype based classifier
predicted_labels = [np.argmin(np.linalg.norm(sample - X_unseen_means, ord = None, axis=1)) + 1 for sample in Xtest]
predicted_labels = np.array(predicted_labels)

# calculating the accuracy
accuracy = np.mean(Ytest.ravel() == predicted_labels)

print("accuracy = ", accuracy)