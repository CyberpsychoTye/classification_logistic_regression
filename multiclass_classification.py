import numpy as np
from typing import Annotated
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self,  number_of_classes, X_train):
        self.rng = np.random.default_rng(169365102679363528484982587533432496514)
        self.classes = number_of_classes
        self.W = self.rng.random((X_train.shape[1], self.classes))
        self.b = self.rng.random((1, self.classes))
        
        
    def stable_softmax(self, logits):
        exponentials = np.exp(logits - np.max(logits, axis = 1).reshape(-1, 1))
        return exponentials/(np.sum(exponentials, axis = 1).reshape(-1, 1))
    
    def predict(self, X_train):
        regression_output = np.dot(X_train, self.W) + self.b
        y_hat = self.stable_softmax(regression_output)
        return np.argmax(y_hat, axis = 1)
    
    def update_parameters(self, new_W, new_b):
        self.W = new_W
        self.b = new_b

    def loss_function(self, y_hat, y_true:Annotated[np.ndarray, "must be one hot encoded"]):
        return -np.mean(y_true * np.log(y_hat))
    
    def one_hot_encode(self, y_true_labels):
        one_hot_encoded = np.eye(len(y_true_labels), self.classes)[y_true_labels]
        return one_hot_encoded

    def fit(self, X_train, y_train, learning_rate = 0.5, epochs = 1000):
        losses = []
        current_epoch = 0
        for i in range(epochs):
            y_hat = self.stable_softmax(np.dot(X_train, self.W) + self.b)
            y_true_hot_encoded = self.one_hot_encode(y_train)
            loss = self.loss_function(y_hat, y_true_hot_encoded)
            mean_gradient_of_loss_regarding_W = (1/X_train.shape[0]) * np.dot(X_train.T, (y_hat - y_true_hot_encoded))
            mean_gradient_of_loss_regarding_b = (1/X_train.shape[0]) * np.sum((y_hat - y_true_hot_encoded), axis = 0)
            new_W = self.W - (learning_rate * mean_gradient_of_loss_regarding_W)
            new_b = self.b - (learning_rate * mean_gradient_of_loss_regarding_b)
            self.update_parameters(new_W , new_b)
            if current_epoch == 0:
                losses.append(loss)
            elif current_epoch % 100 == 0:
                losses.append(loss)
            current_epoch += 1
        
        return losses
    
    def accuracy_score(self, y_hat:np.ndarray, y_true:np.ndarray):
        return np.sum(y_hat == y_true)/len(y_true)
    
X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

classifier = LogisticRegression(3, X_train)
check = classifier.fit(X_train, y_train)
predictions = classifier.predict(X_validate)
print((classifier.accuracy_score(predictions, y_validate)))
print(predictions, y_validate)

confusion_mat = confusion_matrix(predictions, y_validate)
confusion_matrix_display = ConfusionMatrixDisplay(confusion_mat, display_labels = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"])
print(confusion_matrix_display.ax_)
# confusion_matrix_display.plot()
# plt.show()