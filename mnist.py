import numpy as np

class NN:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def forward(self, X):
        # want (b x i) * (i x h) = (b x h)
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.output = self.softmax(self.z2)
        return self.output
    
    def backward(self, X, y):
        # backprop
        dC_dz2 = self.output - y
        # want (h x b) * (b x o) = (h x o)
        dC_dw2 = np.dot(self.a1.T, dC_dz2)

        dC_da1 = np.dot(dC_dz2, self.weights2.T)
        da_dz1 = self.d_relu(self.a1)
        dC_dz1 = dC_da1 * da_dz1
        # want (i x b) * (b x h) = (i x h)
        dC_dw1 = np.dot(X.T, dC_dz1)

        # update weights
        self.weights1 -= self.lr * dC_dw1
        self.weights2 -= self.lr * dC_dw2
        self.bias1 -= self.lr * dC_dz1
        self.bias2 -= self.lr * dC_dz2

def ce_loss(y_pred, y_act):
    # with gradient clipping
    return -1/len(y_act) * np.sum(y_act * np.log(y_pred.clip(min=1e-10)))
    