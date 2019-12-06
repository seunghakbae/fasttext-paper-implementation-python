from Data import Data
from datetime import datetime

def test_model(model, test_path, gram2index, classes):

    test_data = Data(test_path, hash=False)

    text = open("result.txt", 'w')
    correct_answer = 0

    print("\nstart testing")

    for index in range(len(test_data.classes)):
        answer = test_data.classes[index]
        input_gram = []

        for gram in test_data.titles[index]:
            if gram in gram2index:
                input_gram.append(gram2index[gram])

        for gram in test_data.descriptions[index]:
            if gram in gram2index:
                input_gram.append(gram2index[gram])

        guess, prob = model.classify(input_gram)
        guess = guess.item()
        content = 'answer: {}, guess: {}, prob: {:.4}'.format(classes.index2class[answer],classes.index2class[guess], prob)
        text.write(content + "\n")

        if answer == guess:
            correct_answer += 1

    print(f"# of correct answer: {correct_answer} / {len(test_data.classes)}")
    print()
    print("Accuracy = {:.4}".format(correct_answer / len(test_data.classes)))

    text.write("Test finished!\n")
    text.write(f"number of correct answer: {correct_answer} / {len(test_data.classes)}\n")
    text.write("\n")
    text.write("Accuracy = {:.4}\n".format(correct_answer / len(test_data.classes)))
    text.close()

