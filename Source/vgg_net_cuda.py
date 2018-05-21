import data_utils
import constants
import torchvision.models
import torch.optim
from torch.utils.data import DataLoader
import torch.nn
from torch.autograd import Variable
import numpy as np
import time


class VggNet:

    def __init__(self, vgg_type='vgg16', num_classes=3, load_from=None):
        self.vgg_type = vgg_type
        self.best_val_acc = 0.0
        self.val_threshold = 0.7
        if load_from is not None:
            # instantiate the model and replace the last layer with 196 classes so that the stanford model can be
            # loaded
            self.model = torchvision.models.vgg16(pretrained=False)
            self.__replace_last_layer(196)
            self.load_model(load_from)
        else:
            if vgg_type == 'vgg16':
                self.model = torchvision.models.vgg16(pretrained=True)
        # remove the last layer of the pre-trained model and replace it with one required for current classification
        self.__replace_last_layer(num_classes)
        # convert the model to cuda
        self.model.cuda()

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

    def get_damaged_diff_lr(self, params):
        """
        takes input a list of params and converts them into a dictionary
        """
        # dict for layer 13, 14
        params_dictfc = {'params': [params[0], params[1], params[2], params[3]], 'lr': 5e-5}
        # dict for the last layer
        params_dictout = {'params': [params[4], params[5]], 'lr': 1e-4}
        return [params_dictfc, params_dictout]

    def get_stanford_diff_lr(self, params):
        # dict for layer 4,5,6,7,8,9
        params_list = []
        for i in range(12):
            params_list.append(params[i])
        params_dict_conv1 = {'params': params_list, 'lr': 5e-6}

        # dict for layer 10,11,12
        params_list = []
        for i in range(12,18):
            params_list.append(params[i])
        params_dict_conv2 = {'params': params_list, 'lr': 1e-5}

        # dict for layer 13, 14
        params_dictfc = {'params': [params[18], params[19], params[20], params[21]], 'lr': 5e-5}
        # dict for the last layer
        params_dictout = {'params': [params[22], params[23]], 'lr': 1e-4}
        return [params_dictfc, params_dictout]

    def fine_tune_train(self, train_dataset, val_dataset, epochs=10, finetune_from=14, save_model=True):
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

        # define the loss function
        loss_func = torch.nn.CrossEntropyLoss()
        # get the parameters that need to be fine-tuned

        tunable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # use layer-wise differential learning rates
        tunable_params = self.get_stanford_diff_lr(tunable_params)
        # tunable_params = self.get_damaged_diff_lr(tunable_params)

        # define the optimizer
        optimizer = torch.optim.Adam(tunable_params, lr=constants.LR_RATE, weight_decay=constants.WEIGHT_DECAY)
        # optimizer = torch.optim.Adam(tunable_params, weight_decay=constants.WEIGHT_DECAY)
        # define the dataloader
        data_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
        # get the train accuracy before the first epoch
        train_acc, epoch_loss = self.check_accuracy(train_dataset)

        for ep in range(epochs):
            ep_start_time = time.time()
            # check accuracies before every epoch
            val_acc, val_loss = self.check_accuracy(val_dataset)
            print 'Accuracy in epoch ', ep, 'Train acc:', train_acc, 'Val acc:', val_acc
            print 'Loss in epoch ', ep, 'Train loss:', epoch_loss, 'Val loss:', val_loss
            # check if the model needs to be saved
            self.save_model(val_acc, save_model)
            epoch_loss = 0.0
            # set the model to train mode
            self.model.train()
            num_corr = 0
            for i, batch_data in enumerate(data_loader):
                optimizer.zero_grad()
                Xi, yi = batch_data
                scores = self.model.forward(Variable(Xi.cuda(), requires_grad=False))
                loss = loss_func(scores, Variable(yi.cuda(), requires_grad=False))
                epoch_loss += loss.cpu().data.numpy()[0]
                # compute training accuracy for this batch
                yi_pred = np.argmax(scores.cpu().data.numpy(), axis=1)
                num_corr += np.sum(yi.numpy() == yi_pred)
                # compute and update the gradients
                loss.backward()
                optimizer.step()
            train_acc = (num_corr*1.0)/len(train_dataset)
            ep_end_time = time.time()
            print 'Epoch Time:', (ep_end_time-ep_start_time)
        val_acc, val_loss = self.check_accuracy(val_dataset)
        # check if the model needs to be saved
        self.save_model(val_acc, save_model)
        print 'Accuracy in epoch ', ep, 'Train acc:', train_acc, 'Val acc:', val_acc
        print 'Loss in epoch ', ep, 'Train loss:', epoch_loss, 'Val loss:', val_loss

    def load_model(self, file_name):
        """
        loads the model params from the file mentioned in input
        :param file_name:
        :return:
        """
        self.model.load_state_dict(torch.load(constants.MODEL_DIR + file_name))

    def save_model(self, val_acc, save_model):
        """
        Saves the model parameters in MODEL_DIR under filename specified in input
        :param file_name:
        :return:
        """
        if not save_model:
            return
        if val_acc < self.val_threshold:
            return
        if val_acc < self.best_val_acc:
            return
        print 'Saving the model'
        file_name = constants.RUN_ID + '_model.pth'
        self.best_val_acc = val_acc
        torch.save(self.model.state_dict(), constants.MODEL_DIR + file_name)

    def check_accuracy(self, dataset):
        """
        Check the accuracy given by the model for input dataset
        :return: Float containing the accuracy
        """
        # start = time.time()
        # define the loss function
        loss_func = torch.nn.CrossEntropyLoss()
        # set the model to eval mode
        self.model.eval()
        data_loader = DataLoader(dataset, batch_size=64)
        num_corr = 0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader):
            Xi, yi = data
            scores = self.model.forward(Variable(Xi.cuda(), requires_grad=False))
            loss = loss_func(scores, Variable(yi.cuda(), requires_grad=False))
            epoch_loss += loss.cpu().data.numpy()[0]
            yi_pred = np.argmax(scores.cpu().data.numpy(), axis=1)
            num_corr += np.sum(yi.numpy() == yi_pred)
        acc = (num_corr * 1.0) / (len(dataset))
        # end = time.time()
        # print 'Accuracy checks completed in', (end-start)
        return acc, epoch_loss


if __name__ == '__main__':
    vgg = VggNet(num_classes=3)
    print 'Starting the training process'
    train_dataset, val_dataset, test_dataset = data_utils.load_images_ImageFolder(
        constants.TRAIN_DIR, constants.VAL_DIR, constants.TEST_DIR)

    # print 'X_train, y_train', utils.check_datasets_size(train_dataset)
    # print 'X_val, y_val', utils.check_datasets_size(val_dataset)
    # print 'X_test, y_test', utils.check_datasets_size(test_dataset)

    print 'RUN_ID:', constants.RUN_ID
    print 'NUM_EPOCHS:', constants.NUM_EPOCHS
    print 'LR_RATE:', constants.LR_RATE
    print 'FINE_TUNE_LAYER:', constants.FINE_TUNE_LAYER
    print 'BATCH_SIZE:', constants.BATCH_SIZE
    print 'WEIGHT_DECAY:', constants.WEIGHT_DECAY

    vgg.fine_tune_train(train_dataset, val_dataset, epochs=constants.NUM_EPOCHS,
                        finetune_from=constants.FINE_TUNE_LAYER, save_model=True)
    test_acc, ls = vgg.check_accuracy(test_dataset)
    print 'Final test acc:', test_acc
