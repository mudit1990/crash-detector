import data_utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from feature_extracter_vgg import FeatureExtracterVgg


def train_classifier():
    lr_model = LogisticRegression(penalty='l2', dual=False, C=1.0, solver='liblinear', n_jobs=-1)
    data = data_utils.get_img_features('../Data/Objects/vgg16_extracted_features_train.pkl',
                                       '../Data/Objects/vgg16_extracted_features_val.pkl',
                                       '../Data/Objects/vgg16_extracted_features_test.pkl')
    X_train, y_train = data['X_train'], data['y_train']
    lr_model.fit(X_train, y_train)
    return lr_model


lr_model = train_classifier()
fe = FeatureExtracterVgg(vgg_type='vgg16')
ft_x, ft_y = fe.extract_features(save_outputs=False)
print lr_model.predict_proba(ft_x)
