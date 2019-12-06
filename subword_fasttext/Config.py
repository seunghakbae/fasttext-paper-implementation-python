import argparse

def config():

    parser = argparse.ArgumentParser(description='fasttext')
    parser.add_argument('classes_path', metavar='classes-path' , type=str, help='file path of class list')
    parser.add_argument('train_path', metavar='train-path', type=str, help='file path of training data (csv)')
    parser.add_argument('test_path', metavar='test-path', type=str, help='file path of test data (csv)')
    parser.add_argument('n_gram', metavar='n-gram', type=str, help='the number of gram')

    return parser.parse_args()