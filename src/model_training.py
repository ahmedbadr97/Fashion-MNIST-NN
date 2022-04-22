import sys

import torch

from src.model import FashionMinstModel
from torch import nn, optim
from datetime import datetime


def test_model(model: FashionMinstModel, test_loader, criterion):
    test_correct = 0
    test_loss = 0.0
    test_size = len(test_loader.dataset)

    with torch.no_grad():
        model.eval()
        forward_cnt = 0.0
        for data, labels in test_loader:
            log_output = model.forward(data)
            loss = criterion(log_output, labels)
            forward_cnt += 1
            test_loss += loss.item()
            _, top_class = torch.exp(log_output).topk(1, dim=1)
            top_class = top_class.view(-1)
            test_correct += (torch.sum(top_class == labels)).item()

        test_accuracy = (test_correct / float(test_size)) * 100.0
        print()
        model.train()
        test_loss /= float(len(test_loader))
    return test_loss, test_accuracy


def train(model: FashionMinstModel, epochs, learning_rate, train_loader, test_loader=None,
          weights_save_pth="../model_weights"):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_size = len(train_loader.dataset)

    train_losses, test_losses = [], []
    epochs_train_accuracy, epochs_test_accuracy = [], []
    min_test_loss = 10000.0
    min_train_loss = 10000.0

    # test the model before training
    test_loss, test_accuracy = test_model(model, test_loader, criterion)
    print(
        f"model testLoss and accuracy before training \nTest Loss:{str(test_loss)[:8]} Test Accuracy:{str(test_accuracy)[:8]}%\n")

    for e in range(epochs):
        train_loss = 0.0
        train_correct = 0

        forward_cnt = 0.0
        # ___________________Train Model_____________________

        for data, labels in train_loader:
            # ___________________feedForward_____________________

            optimizer.zero_grad()
            log_output = model.forward(data)
            loss = criterion(log_output, labels)
            forward_cnt += 1

            # ___________________backpropagation_____________________

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # _____________ calculate correct output from feed batches ____________
            _, top_class = torch.exp(log_output).topk(1, dim=1)
            top_class = top_class.view(-1)
            train_correct += (torch.sum(top_class == labels)).item()

        # ___________________Test Model_____________________
        test_loss, test_accuracy = test_model(model, test_loader, criterion)

        # ____________________ calculate losses and accuracy___________________
        train_loss /= float(len(train_loader))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracy = (train_correct / float(train_size)) * 100.0
        epochs_train_accuracy.append(train_accuracy)

        epochs_test_accuracy.append(test_accuracy)

        # ____________________ print Training data ___________________

        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.8f}.. ".format(train_loss),
              "Test Loss: {:.8f}.. ".format(test_loss),
              "Train Accuracy: {:.5f}%".format(train_accuracy),
              "Test Accuracy: {:.5f}%".format(test_accuracy)
              )

        # ________________ saving model weights____________________
        # if new score in test accuracy achieved save model weights with file name and time and train_acc and test_acc
        if min_test_loss > test_loss and min_train_loss > train_loss:
            time = datetime.now().strftime("%d_%m %I_%M")
            file_name = f"{time} train_{int(train_accuracy)} test_{int(test_accuracy)}.pth"
            full_pth = f"{weights_save_pth}/{file_name}"
            print(f"new min test and train loss  achieved --> model weights saved in '{weights_save_pth}/{file_name}'")
            torch.save(model.state_dict(), full_pth)
            min_test_loss = test_loss
            min_train_loss = train_loss

        # ___________overfitting test___________#
        if test_loss > train_loss:
            print("!!!Warning overfitting!!!")
        print()

    return train_losses, test_losses, epochs_train_accuracy, epochs_test_accuracy
