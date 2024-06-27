import pickle
from mnist import NN, ce_loss
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def data_split(data, labels, valid_amt=0.1, test_amt=0.1):
    num = data.shape[0]

    split_shape = int(num * test_amt) + int(num * valid_amt)
    valid_shape = int(num * valid_amt)

    X_train, X_testvalid = data[:-split_shape]/255, data[-split_shape:]/255
    X_valid, X_test = X_testvalid[:-valid_shape], X_testvalid[-valid_shape:]
    y_train, y_testvalid = labels[:-split_shape], labels[-split_shape:]
    y_valid, y_test = y_testvalid[:-valid_shape], y_testvalid[-valid_shape:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train(X, y, model):
    y_pred = model.forward(X)
    loss = ce_loss(y_pred, y)
    model.backward(X, y)
    return loss

def train_epoch(X_train, y_train, model):
    total_loss = 0
    for i in range(len(X_train)):
        one_hot = np.zeros(10)
        one_hot[y_train[i]] = 1
        X = X_train[i].reshape(1, -1)
        total_loss += train(X, one_hot, model)
    return total_loss / len(X_train)

def evaluate(X_valid, y_valid, model):
    correct = 0
    for i in range(len(X_valid)):
        X = X_valid[i].reshape(1, -1)
        output = model.forward(X)
        if np.argmax(output) == y_valid[i]:
            correct += 1
    return correct / len(X_valid)

def main():
    file = 'data/MNIST-120k'
    file_data = unpickle(file)
    data = file_data['data'].reshape(-1, 784)
    labels = file_data['labels']

    X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(data, labels)
    
    BATCH_SIZE = 32
    EPOCHS = 10
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    lr = 1e-2
    
    model = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, lr)
    for epoch in range(EPOCHS):
        total_loss = train_epoch(X_train, y_train, model)
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
        acc = evaluate(X_valid, y_valid, model)
        print(f'Validation Accuracy: {acc}')
    
    test_acc = evaluate(X_test, y_test, model)
    print(f'Final Test Accuracy: {test_acc}')

    np.save('w1.npy', model.weights1)
    np.save('w2.npy', model.weights2)
    np.save('b1.npy', model.bias1)
    np.save('b2.npy', model.bias2)

if __name__ == '__main__':
    main()