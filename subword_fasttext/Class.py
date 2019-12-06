import csv

class Class:

    def __init__(self, class_path):

        file = open(class_path, 'r')
        data = file.read()

        self.classes_list = data.split()

        self.class2index = {}
        self.index2class = {}

        for index, classes in enumerate(self.classes_list):
            self.class2index[classes] = index
            self.index2class[index] = classes
