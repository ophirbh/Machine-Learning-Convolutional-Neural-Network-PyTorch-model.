from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from gcommand_loader import GCommandLoader
from convolutional_nn import ConvolutionalNN
from neural_net import NeuralNet

# Hyperparameters
ETA = 0.001
BATCH_SIZE = 100
EPOCH_NUM = 4


def train(model, loader, optimizer, cuda, loging=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target, path) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data

        if loging and batch_idx % 5 == 0:
            print(batch_idx)

    return global_epoch_loss / len(loader.dataset)


def test(model, loader, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target, path) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # print_label_vs_predict(model, data, target)

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss


def main():

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    #train_set = GCommandLoader('./ML4_dataset/data/train')
    #validation_set = GCommandLoader('./ML4_dataset/data/valid')
    #test_set = GCommandLoader('./ML4_dataset/data/test')
    train_set = GCommandLoader('./data/train')
    validation_set = GCommandLoader('./data/valid')
    test_set = GCommandLoader('./data/test')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None) #bla

    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None) # bla

    #model = NeuralNet(101 * 161)
    model = ConvolutionalNN().to(device)
    #model = ConvolutionalNN()

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ETA)
    for epoch in range(EPOCH_NUM):
        print("epoch " + str(epoch))
        train(model, train_loader, optimizer, cuda)  # bla
        test(model, validation_loader, cuda)
    print_report(model, test_loader, cuda)


def print_label_vs_predict(model, data, label):
    predict = model.forward(data)
    for index in range(data.shape[0]):
        values, indices = torch.max(predict[index], 0)
        print("label-" + str(int(label[index])) + ":predict-" + str(int(indices)))


def print_report(model, data, cuda):
    output_file = open("test_y", "w+")

    for batch_idx, (data, target, path) in enumerate(data):
        if cuda:
            data = data.cuda()
        data = Variable(data)
        predict = model.forward(data)
        for i in range(predict.size()[0]):
            path_to_print = str(path[i]).split("/")[-1]
            values, indices = torch.max(predict[i], 0)
            output_file.write(path_to_print + ", " + str(int(indices)) + "\n")
            print(path_to_print + ", " + str(int(indices)) + "\n")

    output_file.close()


main()
