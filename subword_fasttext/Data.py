import csv
import re

class Data:

    def __init__(self, train_path,hash=True, n_gram=2):

        print("loading train data...")

        file = open(train_path, 'r')
        csv_reader =csv.reader(file)

        self.classes = []
        self.titles = []
        self.descriptions = []

        for line in csv_reader:
            # class of data
            self.classes.append(int(line[0]) - 1)
            # print(int(line[0]) - 1)

            # title of data
            self.titles.append(self.get_word_grams(self.preprocess(line[1])))
            # print(self.get_word_grams(self.preprocess(line[1])), n_gram)
            # description of data
            self.descriptions.append(self.get_word_grams(self.preprocess(line[2])))
            # print(self.get_word_grams(self.preprocess(line[2])), n_gram)

        file.close()

        print("finished loading")
        print("loaded news :" + str(len(self.classes)))

        if hash == True:

            print("splitting into n-grams and hashing...")
            self.grams_hash = {}  # dictionary between grams and hash
            self.total_hash = set()  # set that contains all hash
            self.grams2hash = {} # dictionary from grams to hash
            self.hash2grams = {} # dictionary from hash to grams

            count = 0

            for title in self.titles:
                for gram in title:
                    self.grams2hash[gram] = self.fnv_hash(gram)

            for desc in self.descriptions:
                for gram in desc:
                    self.grams2hash[gram] = self.fnv_hash(gram)


            self.total_hash = set([])
            for _, hash_val in self.grams2hash.items():
                self.total_hash.add(hash_val)

            self.hash2index = {}
            for index, hash_val in enumerate(self.total_hash):
                self.hash2index[hash_val] = index

            self.gram2index = {}
            for gram,hash_val in self.grams2hash.items():
                self.gram2index[gram] = self.hash2index[hash_val]

            self.input_seq = []
            self.output_seq = []
            for index in range(len(self.classes)):
                self.input_seq.append(
                    [self.gram2index[gram] for gram in self.titles[index] + self.descriptions[index]]
                )
                self.output_seq.append(self.classes[index])

            print("splitting and hashing done")
            print("total hashes number : " + str(len(self.total_hash)))

    def fnv_hash(self, str, k=2100000):

        fnv_offset_basis = 0xcbf29ce484222325 # fnv offset basis
        # fnv_offset_basis = 0x811c9dc5
        fnv_prime = 0x100000001b3 # fnv_prime

        hash = fnv_offset_basis
        for s in str: # for each character
            hash = hash ^ ord(s) # XOR
            hash *= fnv_prime # multiply by fnv_prime
            hash = hash % k

        return hash

    def preprocess(self, str):

        str = re.sub("[^a-zA-Z]", " ",str)

        return str.lower()

    # Apply word n-grams to string
    def get_word_grams(self, str, n_gram=2):

        list_of_words = str.split()
        grams_list = []

        for i in range(len(list_of_words) - n_gram + 1):
            grams_list.append('-'.join(list_of_words[i:i+n_gram]))

        return grams_list

