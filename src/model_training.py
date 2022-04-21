import torch

from src.model import FashionMinstModel
from torch import nn, optim


def train(model: FashionMinstModel, epochs, learning_rate, train_loader, test_loader=None):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    train_losses, test_losses = [], []

    for e in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        train_correct = 0
        test_correct = 0

        for data, label in train_loader:
            optimizer.zero_grad()
            log_output = model.forward(data)
            loss = criterion(log_output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, top_class = torch.exp(log_output).topk(1, dim=1)
            top_class=top_class.view(-1)
            train_correct += torch.sum(top_class == label)
        train_losses.append(train_loss / len(train_loader))

        with torch.no_grad():
            model.eval()
            for data, label in test_loader:
                log_output = model.forward(data)
                loss = criterion(log_output, label)

                test_loss += loss.item()
                _, top_class = torch.exp(log_output).topk(1, dim=1)
                top_class = top_class.view(-1)
                test_correct += torch.sum(top_class == label)
        test_losses.append(test_loss/len(test_loader))
        model.train()

        train_accuracy = (train_correct / float(train_size)) * 100.0
        test_accuracy = (test_correct / float(test_size)) * 100.0
        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.8f}.. ".format(train_loss / train_size),
              "Test Loss: {:.8f}.. ".format(test_loss / test_size),
              "Train Accuracy: {:.5f}".format(train_accuracy),
              "Test Accuracy: {:.5f}".format(test_accuracy)
              )
        if train_accuracy > test_accuracy:
            print("_________________Warning overfit____________________")
    return train_losses, test_losses