import time
from os.path import exists

import numpy as np
import sklearn
import torch

from torch import nn
from torch.utils.data import DataLoader

from model import CatsVSDogs, Classifier

from sklearn.metrics import confusion_matrix, f1_score


def test_model1(model1, dataloader, device):
    final_loader = DataLoader(dataset=dataloader, batch_size=len(dataloader.data), shuffle=True)
    val_features, val_labels = next(iter(final_loader))
    print(f"Feature batch shape: {val_features.size()}")
    print("IDK WHY IT DOESN'T WORK AFTER THIS")
    final_y = model1(val_features).to(device)
    print("F1 Score: ")
    print(f1_score(final_y, val_labels, average='weighted'))


def test_model(model, dataloader, device):
    CM = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            valx, valy = data[0].to(device), data[1].to(device)
            val_pred = model(valx)
            preds = torch.argmax(val_pred.data, 1)
            CM += confusion_matrix(valy.cpu(), preds.cpu(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        tn = CM[0][0]
        tp = CM[1][1]
        fp = CM[0][1]
        fn = CM[1][0]
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matrix : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)
        print('- Precision: ', (tp / (tp + fp)) * 100)
        print('- NPV: ', (tn / (tn + fn)) * 100)
        print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
        print()

    return acc, CM


if __name__ == '__main__':

    catsvdogs = CatsVSDogs()
    test_set = catsvdogs.train(False)
    train_set = catsvdogs.train(True)
    print("Number of examples :")
    print(len(train_set.data))
    print("Size of each example: ")
    print(np.shape(train_set.data[0]))
    print("Number of testing data")
    print(len(test_set.data))

    # setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if not exists("fashion_28.pth"):
        print("Model does not exist... training...")
        model = Classifier().to(device)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        # device = 'cpu'
        # print(device)
        cirection = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # model.eval()

        # train
        # model.train()
        # when 40 epochs acc reach 94-95%
        # epochs setting 1 just for run
        epochs = 1
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_acc = 0.0
            val_acc = 0.0
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = data[0].to(device), data[1].to(device)
                y_pred = model(x)
                loss = cirection(y_pred, y.long())
                loss.backward()
                optimizer.step()
                train_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.cpu().numpy())
                train_loss += loss.item()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    valx, valy = data[0].to(device), data[1].to(device)
                    val_pred = model(valx)
                    batch_loss = cirection(val_pred, valy.long())
                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == valy.cpu().numpy())
                    val_loss += batch_loss.item()

                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                      (epoch + 1, epochs, time.time() - epoch_start_time, \
                       train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / test_set.__len__(),
                       val_loss / test_set.__len__()))

        torch.save(model, 'fashion_28.pth')

    else:
        print("Model available... loading...")
        # Loading the saved model
        model = torch.load("fashion_28.pth")
        test_model1(model, test_set, device)
