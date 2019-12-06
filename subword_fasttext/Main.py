from Config import config
from Data import Data
from Class import Class
from Train import Fasttext_trainer
from Test import test_model

def main():
    args = config()
    class_path = args.classes_path # classes_path
    train_path = args.train_path # train_path
    test_path = args.test_path # test_path
    n_gram = args.n_gram # ngram

    data = Data(train_path, hash=True, n_gram=n_gram)
    classes = Class(class_path)
    model = Fasttext_trainer(data, classes, dimension=300, learning_rate=0.05, epoches=10)
    test_model(model, test_path, data.gram2index, classes)

main()