import data_utils
import utils
import constants
import torchvision.models
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn
from torch.autograd import Variable
import numpy as np
import time


class VggNet:

    def __init__(self, vgg_type='vgg16', num_classes=3, load_from=None):
        self.vgg_type = vgg_type
        if load_from is not None:
            self.load_model(load_from)
        else:
            if vgg_type == 'vgg16':
                self.model = torchvision.models.vgg16(pretrained=True)
        # remove the last layer of the pre-trained model and replace it with one required for current classification
        self.__replace_last_layer(num_classes)

    def __replace_last_layer(self, num_classes):
        """
        replaces the last layer from a 1000 way classifiers to a 3 way classifier needed for the task
        :return:
        """
        new_classifier = list(self.model.classifier)
        # replace the last layer
        new_classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        # convert to sequential object and put it back in the model
        self.model.classifier = torch.nn.Sequential(*new_classifier)

    def fine_tune_train(self, train_dataset, val_dataset, epochs=10, finetune_from=12, save_model=True):
        """
        :param train_dataset: A Dataset object for train data that can be directly used in Dataloader
        :param X_val: validation X data
        :param y_val: validation y data
        :param epochs: number of iterations over the entire data for training
        :param finetune_from: This and the layers after it will have requires_grad set to True and will be finetuned
        :return:
        """
        # set all params requires_grad before finetune_from as false
        l = 0
        for params in self.model.parameters():
            if l < 2 * finetune_from:
                params.requires_grad = False
            else:
                params.requires_grad = True
            l += 1

        # for params in self.model.parameters():
        #     print params.requires_grad
        # print 'now the classifier'
        # for params in self.model.classifier.parameters():
        #     print params.requires_grad

        # define the loss function
        loss_func = torch.nn.CrossEntropyLoss()
        # get the parameters that need to be fine-tuned
        tunable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # define the optimizer
        optimizer = torch.optim.Adam(tunable_params, lr=constants.LR_RATE)
        # define the dataloader
        data_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
        train_acc, epoch_loss = self.check_accuracy(train_dataset)

        for ep in range(epochs):
            epoch_loss = 0.0
            # check accuracies before every epoch
            val_acc, val_loss = self.check_accuracy(val_dataset)
            print 'Accuracy in epoch ', ep, 'Train acc', train_acc, 'Val acc', val_acc
            print 'Loss in epoch ', ep, 'Train loss:', epoch_loss, 'Val loss:', val_loss
            # set the model to train mode
            self.model.train()
            num_corr = 0
            for i, batch_data in enumerate(data_loader):
                optimizer.zero_grad()
                Xi, yi = batch_data
                start = time.time()
                scores = self.model.forward(Variable(Xi, requires_grad=False))
                end = time.time()
                print 'Forward pass for iteration', i, (end - start)
                loss = loss_func(scores, Variable(yi, requires_grad=False))
                epoch_loss += loss.data.numpy()[0]
                # compute training accuracy for this batch
                yi_pred = np.argmax(scores.data.numpy(), axis=1)
                num_corr += np.sum(yi.numpy() == yi_pred)
                # compute and update the gradients
                start = time.time()
                loss.backward()
                optimizer.step()
                end = time.time()
                print 'Backward pass for iteration', i, (end - start)
            train_acc = (num_corr * 1.0) / len(train_dataset)
        val_acc = self.check_accuracy(val_dataset)
        print 'Accuracy in epoch ', ep, 'Train acc:', train_acc, 'Val acc:', val_acc
        print 'Loss in epoch ', ep, 'Train loss:', epoch_loss, 'Val loss:', val_loss
        if save_model:
            file_name = constants.RUN_ID + '_model.pth'
            self.save_model(file_name)

    def load_model(self, file_name):
        """
        loads the model params from the file mentioned in input
        :param file_name:
        :return:
        """
        self.model.load_state_dict(torch.load(constants.MODEL_DIR + file_name))

    def save_model(self, file_name):
        """
        Saves the model parameters in MODEL_DIR under filename specified in input
        :param file_name:
        :return:
        """
        torch.save(self.model.state_dict(), constants.MODEL_DIR + file_name)

    def check_accuracy(self, dataset):
        """
        Check the accuracy given by the model for input dataset
        :return: Float containing the accuracy
        """
        # define the loss function
        loss_func = torch.nn.CrossEntropyLoss()
        # set the model to eval mode
        self.model.eval()
        data_loader = DataLoader(dataset, batch_size=64)
        num_corr = 0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader):
            Xi, yi = data
            start = time.time()
            scores = self.model.forward(Variable(Xi, requires_grad=False))
            end = time.time()
            print 'Forward pass for accuracy iteration', i, (end - start)
            loss = loss_func(scores, Variable(yi, requires_grad=False))
            epoch_loss += loss.data.numpy()[0]
            yi_pred = np.argmax(scores.data.numpy(), axis=1)
            num_corr += np.sum(yi.numpy() == yi_pred)
        acc = (num_corr*1.0)/(len(dataset))
        return acc, epoch_loss


if __name__ == '__main__':
    vgg = VggNet(num_classes=196)
    print 'Starting the training process'
    train_dataset, val_dataset, test_dataset = data_utils.load_images_ImageFolder(
        constants.TRAIN_STANFORD_DIR, constants.VAL_STANFORD_DIR, constants.TEST_STANFORD_DIR)

    print 'X_train, y_train', utils.check_datasets_size(train_dataset)
    print 'X_val, y_val', utils.check_datasets_size(val_dataset)
    print 'X_test, y_test', utils.check_datasets_size(test_dataset)

    print 'RUN_ID:', constants.RUN_ID
    print 'NUM_EPOCHS:', constants.NUM_EPOCHS
    print 'LR_RATE:', constants.LR_RATE
    print 'FINE_TUNE_LAYER:', constants.FINE_TUNE_LAYER
    print 'BATCH_SIZE:', constants.BATCH_SIZE
    print 'WEIGHT_DECAY:', constants.WEIGHT_DECAY

    vgg.fine_tune_train(train_dataset, val_dataset, epochs=constants.NUM_EPOCHS,
                        finetune_from=constants.FINE_TUNE_LAYER, save_model=False)
    test_acc = vgg.check_accuracy(test_dataset)
    print 'Final test acc:', test_acc
