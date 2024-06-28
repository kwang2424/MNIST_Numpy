import argparse

parser = argparse.ArgumentParser(description='Numpy MNIST implementation')
parser.add_argument('-f', '--file', type=str, default='data/MNIST-120k', help='Path to MNIST data')
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2, help='Learning rate')
args = parser.parse_args()

FILE = args.file
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
lr = args.learning_rate