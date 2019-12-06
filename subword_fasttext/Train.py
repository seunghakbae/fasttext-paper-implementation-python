import torch

class Fasttext_trainer:

    def __init__(self, data, classes, dimension=300, learning_rate=0.05, epoches=10):
        # Xavier initialization of weight matrices
        input_size = len(data.total_hash)
        hidden_size = dimension
        output_size = len(classes.classes_list)

        self.w_in = torch.randn(input_size, hidden_size) / (hidden_size ** 0.5)
        self.w_out = torch.randn(hidden_size, output_size) / (hidden_size ** 0.5)

        count = 0
        losses = []
        for epoch in range(epoches):
            print()
            print(f"Epoch #{epoch + 1}")
            print("Losses: ")

            for inputs, output in zip(data.input_seq, data.output_seq):
                count += 1

                loss = self.train(inputs, output, learning_rate)
                losses.append(loss.item())

                if count % 5000 == 0:
                    avg_loss = sum(losses) / len(losses)
                    print("%f" % (avg_loss), end=", ", flush=True)
                    losses = []

            print()
            print("\nTraining finished!")

    def train(self, inputs, output, learning_rate):

        input_size = len(inputs)
        hidden = torch.sum(self.w_in[inputs], dim=0) / input_size
        scores = torch.squeeze(torch.mm(hidden.view(1, -1), self.w_out))

        e_scores = torch.exp(scores)
        softmax = e_scores / torch.sum(e_scores)

        loss = -torch.log(softmax[output])

        softmax[output] -= 1

        grad_out = torch.mm(
            hidden.view(-1, 1), # (D,1)
            softmax.view(1, -1) # (1,C)
        )  # (D,C)

        grad_in = torch.mm(
            self.w_out, # (D,C)
            softmax.view(-1, 1) # (C,1)
        ).t() / input_size # (1,D)

        # Update weight matrices
        self.w_in[inputs] -= learning_rate * grad_in.squeeze()
        self.w_out -= learning_rate * grad_out

        return loss

    def classify(self, inputs_seq):
        input_size = len(inputs_seq)

        hidden = torch.sum(self.w_in[inputs_seq], dim=0) / input_size  # (D)

        scores = torch.squeeze(torch.mm(hidden.view(1, -1), self.w_out)) # (C)

        e_scores = torch.exp(scores)
        softmax = e_scores / torch.sum(e_scores)

        prob, index = softmax.max(dim=0)
        return index, prob
