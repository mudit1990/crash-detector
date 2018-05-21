import torchvision.models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import data_utils
from torch.autograd import Variable
import pickle
import os
import time


class FeatureExtracterVgg:

    def __init__(self, vgg_type='vgg16'):
        self.features = None
        self.vgg_type = vgg_type
        if vgg_type == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=True)
        # set all the parameter grads to false
        for param in self.model.parameters():
            param.requires_grad=False
        # set the model to eval phase
        self.model.eval()
        # extract the vgg classifier and register the forward hook
        vgg_classifier = self.model.classifier
        vgg_classifier[5].register_forward_hook(self.__copy_output)

    def __copy_output(self, model, input, output):
        print 'the forward hook was called'
        data = output.data.numpy()
        self.features = np.copy(data)

    def extract_features(self, save_outputs=True):
        """
        loads all the images as tensor objects, runs them through the model. The model extracts the output of
        the last fc layer before outputs and sets it in self.features. The method then outputs self.features, y as a
        pickle object
        :return:
        """
        # X, y = data_utils.load_imgs('../OriginalData/car-damage-dataset/data1a/occlusion_exp/')
        X, y = data_utils.load_imgs_damaged_good('../Data/Images/Damage_Good/Good/',
                                                 '../Data/Images/Damage_Good/Damaged/')
        # X, y = data_utils.load_damage_classification_imgs(
        #     minor_dir='../Data/Images/Damage_Classification_Extended_V2/test/01-minor',
        #     moderate_dir='../Data/Images/Damage_Classification_Extended_V2/test/02-moderate',
        #     major_dir='../Data/Images/Damage_Classification_Extended_V2/test/03-severe')
        print 'input shape', X.shape, y.shape
        # creating a tensor dataset to use in dataloader
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        out_X = None
        out_y = None
        for i, batch_data in enumerate(data_loader):
            Xi, yi = batch_data
            print 'Starting forward pass for iteration', i
            start = time.time()
            self.model.forward(Variable(Xi, requires_grad=False))
            out_X, out_y = self.save_batch_output(self.features, yi.numpy(), out_X, out_y)
            end = time.time()
            print 'time taken secs:', (end-start)

        if save_outputs:
            out_file = self.vgg_type + '_features_damaged_good.pkl'
            out_data = (out_X, out_y)
            with open(os.path.join('../Data/Objects/', out_file), 'wb') as f:
                pickle.dump(out_data, f)
        print 'output shape', out_X.shape, out_y.shape
        return out_X, out_y

    def save_batch_output(self, in_x, in_y, out_x, out_y):
        if out_x is None or out_y is None:
            return in_x, in_y
        out_x = np.concatenate((out_x, in_x))
        out_y = np.concatenate((out_y, in_y))
        return out_x, out_y


fe = FeatureExtracterVgg(vgg_type='vgg16')
fe.extract_features(save_outputs=True)