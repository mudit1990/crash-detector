import data_utils
import constants
import torchvision.models
import torch.optim
from torch.utils.data import DataLoader
import torch.nn
from torch.autograd import Variable
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os


class Model_Ensembles:

    def __init__(self, load_from):
        """
        Takes input a list of filenames from which a model needs to be loaded and instantiates the models from the
        files
        """
        self.models = []
        for fl in load_from:
            model = torchvision.models.vgg16(pretrained=False)
            self.__replace_last_layer(model, 3)
            model.load_state_dict(torch.load(constants.MODEL_DIR + fl))
            model.cuda()
            self.models.append(model)

    def __replace_last_layer(self, model, num_classes):
        """
        replaces the last layer from a 1000 way classifiers to a 3 way classifier needed for the task
        :return:
        """
        new_classifier = list(model.classifier)
        # replace the last layer
        new_classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        # convert to sequential object and put it back in the model
        model.classifier = torch.nn.Sequential(*new_classifier)

    def get_accuracy(self, dataset, generate_report=False):
        """
        check the accuracy for given dataset on ensemble of models
        """
        softmax = torch.nn.Softmax()
        data_loader = DataLoader(dataset, batch_size=64)
        num_corr = 0
        y, y_pred = np.array([]), np.array([])
        for i, data in enumerate(data_loader):
            Xi, yi = data
            y = np.append(y, yi)
            probs = np.zeros((len(yi), 3))
            for model in self.models:
                model.eval()
                scores = model.forward(Variable(Xi.cuda(), requires_grad=False))
                model_prob = softmax(scores)
                probs += model_prob.cpu().data.numpy()
            probs /= len(self.models)
            yi_pred = np.argmax(probs, axis=1)
            y_pred = np.append(y_pred, yi_pred)
            num_corr += np.sum(yi.numpy() == yi_pred)
        acc = (num_corr * 1.0) / (len(dataset))
        if generate_report:
            print classification_report(y, y_pred)
            print confusion_matrix(y, y_pred)
        return acc, y, y_pred


if __name__ == '__main__':
    ensembles = Model_Ensembles(load_from=[])
    train_dataset, val_dataset, test_dataset = data_utils.load_images_ImageFolder(
        constants.TRAIN_DIR, constants.VAL_DIR, constants.TEST_DIR)
    val_ensemble_acc, valy, valy_pred = ensembles.get_accuracy(val_dataset, generate_report=True)
    test_ensemble_acc, testy, testy_pred = ensembles.get_accuracy(test_dataset, generate_report=True)
    print 'val_ensemble_acc:', val_ensemble_acc
    print 'test_ensemble_acc', test_ensemble_acc
    # save pred, actual, images as pickle file
    filename = 'ensemble_results'
    filepath = os.path.join('../Models', filename)
    data = (test_dataset.imgs, testy, testy_pred)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
