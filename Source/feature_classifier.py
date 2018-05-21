import data_utils
import utils
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


def train_classifier(model, data):
    X_train, y_train = data['X_train'], data['y_train']
    print 'training size', X_train.shape
    model.fit(X_train, y_train)
    return model


def get_performance(model, data):
    X_train, y_train = data['X_train'], data['y_train']
    y_train_pred = model.predict(X_train)
    print 'Train size', X_train.shape
    print 'Train Acc', np.mean(y_train == y_train_pred)
    print 'Train Matrix', confusion_matrix(y_train, y_train_pred)

    X_val, y_val = data['X_val'], data['y_val']
    y_val_pred = model.predict(X_val)
    print 'Validation size', X_val.shape
    print 'Validation Acc', np.mean(y_val == y_val_pred)
    print 'Validation Matrix', confusion_matrix(y_val, y_val_pred)

    X_test, y_test = data['X_test'], data['y_test']
    y_test_pred = model.predict(X_test)
    print 'Test size', X_test.shape
    print 'Test Acc', np.mean(y_test == y_test_pred)
    print 'Test Matrix', confusion_matrix(y_test, y_test_pred)

    print classification_report(y_test, y_test_pred)


# data = data_utils.get_img_features('../Data/Objects/vgg16_extracted_features_train.pkl',
#                           '../Data/Objects/vgg16_extracted_features_val.pkl',
#                           '../Data/Objects/vgg16_extracted_features_val.pkl')
#
# utils.plot_tsne(data['X_train'], data['y_train'], 't-SNE plot for features, Damage Recognition')

data = data_utils.get_img_features_split('../Data/Objects/vgg16_features_damaged_good.pkl')
model = SVC()
model = train_classifier(model, data)
get_performance(model, data)

# data = data_utils.get_img_features('../Data/Objects/vgg16_features_classification_train.pkl',
#                           '../Data/Objects/vgg16_features_classification_val.pkl',
#                           '../Data/Objects/vgg16_features_classification_test.pkl')
#
# utils.plot_tsne(data['X_train'], data['y_train'], 't-SNE plot for features, Damage Classification')

# data = data_utils.load_raw_images()

# model = LogisticRegression(penalty='l2', dual=False, C=1e-5, solver='liblinear', n_jobs=-1)
# model = LogisticRegression()
# a 67.84% classification accuracy for extracted features
# model = SVC(C=1, class_weight='balanced')
# model = train_classifier(model, data)
# get_performance(model, data)


############################ Results #############################
##### classification accuracy with extracted features = %67.84
##### classification accuracy with raw images = %39.18
## notes: These accuracies were measured with SVC, rbf kernal, checked for overfit and used the balanced option
##################################################################

