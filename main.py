import pickle
from mnist import NN, ce_loss
import numpy as np
import config

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
    imperfect = False
    for i in range(len(X_train) // config.BATCH_SIZE):
        if i + config.BATCH_SIZE < len(X_train):
            one_hot = np.zeros((config.BATCH_SIZE, 10))
            X = X_train[i:i+config.BATCH_SIZE].reshape(config.BATCH_SIZE, -1)
        else:
            # one_hot = np.zeros((len(X_train) - i, 10)) 
            # X = X_train[i:].reshape(len(X_train) - i, -1)
            imperfect = len(X_train) - i
            break
        for j in range(i, i+len(one_hot)):
            one_hot[j-i][y_train[j]] = 1
        # try:
        #     X = X.reshape(config.BATCH_SIZE, -1)
        # except:
        #     print(X.shape, i, i+config.BATCH_SIZE, len(X_train))
        total_loss += train(X, one_hot, model)
    length = len(X_train) - imperfect if imperfect else len(X_train) 
    return total_loss / length

def evaluate(X_valid, y_valid, model):
    correct = 0
    imperfect = 0
    for i in range(len(X_valid) // config.BATCH_SIZE):
        if i + config.BATCH_SIZE >= len(X_valid):
            imperfect = len(X_valid) -  i 
            break
        X = X_valid[i:i+config.BATCH_SIZE]
        y = y_valid[i:i+config.BATCH_SIZE].reshape(-1)
        X = X.reshape(config.BATCH_SIZE, -1)
        output = model.forward(X)
        amt = sum(np.argmax(output, axis=1) == y)
        # print(np.argmax(output, axis=1), y, (np.argmax(output, axis=1) == y).shape, np.argmax(output, axis=1).shape, y.shape, amt)
        # break
        correct += amt
        # print(amt >= 32, i)
    length = len(X_valid) - imperfect if imperfect else len(X_valid) 
    return correct / length

def main():
    file = 'data/MNIST-120k'
    file_data = unpickle(file)
    data = file_data['data'].reshape(-1, 784)
    labels = file_data['labels']

    X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(data, labels)
    
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    
    model = NN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, config.lr, config.BATCH_SIZE)
    for epoch in range(config.EPOCHS):
        total_loss = train_epoch(X_train, y_train, model)
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
        acc = evaluate(X_valid, y_valid, model)
        print(f'Validation Accuracy: {acc}')
    
    test_acc = evaluate(X_test, y_test, model)
    print(f'Final Test Accuracy: {test_acc}')

    np.save('w11.npy', model.weights1)
    np.save('w21.npy', model.weights2)
    np.save('b11.npy', model.bias1)
    np.save('b21.npy', model.bias2)

if __name__ == '__main__':
    main()