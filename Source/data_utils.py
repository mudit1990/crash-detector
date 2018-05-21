import os
from PIL import Image
import scipy.io
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import utils
import constants

resize_preprocess_imgs = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

resize_small_images = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


def save_damaged_car_images(inp_dirs, out_dir):
    """
    reads all the image files present in all the inp_dirs, resizes them to 224x224 and saves them in out_dir by
    numbering the images starting from 0000.
    :param inp_dirs:
    :param out_dir:
    :return: None
    """
    print_after_iter = 100
    file_num = 0
    for dir in inp_dirs:
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for old_filename in files:
            ext = os.path.splitext(old_filename)[1]
            new_filename = 'image_' + format(file_num, "04") + ext
            file_num += 1
            src = os.path.join(dir, old_filename)
            dst = os.path.join(out_dir, new_filename)
            img = Image.open(src).resize((224, 224))
            img.save(dst)
            if file_num%print_after_iter == 0:
                print file_num, 'files resized!'


def load_stanford_annotations(org_meta_file, processed_meta_file):
    """
    The function loads all the annotations from the original .mat annotations files (given by org_meta_file), processes,
     extracts the image_name, class_id and class_name and stores it as a pickle file (processed_meta_file)
    """
    meta_data = scipy.io.loadmat(org_meta_file)
    car_labels = meta_data['class_names'].reshape(-1)
    car_label_dict = {}
    for i in range(len(car_labels)):
        label = car_labels[i][0].astype(str)
        car_label_dict[i+1] = label
    annots = meta_data['annotations'].reshape(-1)
    data = []
    for an in annots:
        # get the image number
        name = an[0][0].astype(str)
        # remove the path
        name = name.split('/')[1]
        clss_id = an[5][0][0]
        clss_name = car_label_dict[clss_id]
        data.append([name, clss_id, clss_name])
    # save as pickle file
    with open(processed_meta_file, 'wb') as f:
        pickle.dump(data, f)


def load_images_astensors(inp_dir, transform_func=resize_preprocess_imgs):
    """
    Reads all the images present in inp_dir, processes them using torch.transforms and returns stacked tensors
    :param inp_dir: Directory path containing the images
    :param: transform_func: The transformation that needs to be applied to raw images to convert them to tensor
    :return: torch.tensor
    """
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    imgs = []
    for f in files:
        if f == '.DS_Store':
            continue
        img_path = os.path.join(inp_dir, f)
        pil_image = Image.open(img_path)
        # ignoring the grey scale images since they are very less
        if utils.is_greyscale(pil_image):
            continue
        tpil = transform_func(pil_image)
        if tpil.shape[0] != 3:
            print tpil.shape
            print img_path
        imgs.append(transform_func(pil_image))
    return torch.stack(imgs)


def load_imgs(dir):
    print 'Loading images...'
    start = time.time()
    X = load_images_astensors(dir)
    end = time.time()
    print 'Images loaded in secs:', (end - start)
    # create dummy y
    y = np.ones(len(X))
    return X, torch.from_numpy(y)


def load_imgs_damaged_good(good_dir, damaged_dir):
    """
    loads the damaged and good images, processes them and returns them as torch.FloatTensor. The labels are encoded as
    1 for damaged and 0 for good. Both X and y are returned as tensors
    :return: X: stacked float tensors of images
    :return: y: labels corresponding to the images
    """
    print 'Loading images...'
    start = time.time()
    dam_X = load_images_astensors(damaged_dir)
    good_X = load_images_astensors(good_dir)
    X = torch.cat([dam_X, good_X], dim=0)
    # damaged is 1
    dam_y = np.ones(len(dam_X))
    # good is 0
    good_y = np.zeros(len(good_X))
    y = np.concatenate((dam_y, good_y))
    end = time.time()
    print 'Images loaded in secs:', (end-start)
    return X, torch.from_numpy(y)


def load_damage_classification_imgs(minor_dir, moderate_dir, major_dir, small_images=False):
    """
    loads and pre-processes the images for damage classification (minor, moderate, severe) and returns images and
    corresponding labels as tensors
    """
    transform_func = resize_small_images if small_images else resize_preprocess_imgs
    minor_X = load_images_astensors(minor_dir, transform_func)
    moderate_X = load_images_astensors(moderate_dir, transform_func)
    major_X = load_images_astensors(major_dir, transform_func)
    X = torch.cat([minor_X, moderate_X, major_X])

    minor_y = np.zeros(len(minor_X))
    moderate_y = np.ones(len(moderate_X))
    major_y = np.ones(len(major_X))+1
    y = np.concatenate((minor_y, moderate_y))
    y = np.concatenate((y, major_y))
    return X, torch.from_numpy(y).type(torch.LongTensor)


def get_img_features_split(inp_path):
    """
    The method loads the pickle file from the given inp_path. It assumes the pickle file to contain X, y where X are
    image features and y are labels. It then splits the features into train, validation and test. The data is first
    split by taking 75% as training. From the remaining 25%, 75% is used as test data and 25% is used as validation
    data. Returns a dictionary with key as 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
    """
    with open(inp_path, 'rb') as f:
        X, y = pickle.load(f)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25, random_state=42, shuffle=True)
    data = {'X_train':X_train, 'y_train':y_train,
            'X_val':X_val, 'y_val':y_val,
            'X_test':X_test, 'y_test':y_test}
    return data


def get_img_features(train_file, val_file, test_file):
    """
    The method loads pre-splitted train, validation and test pickle files and combines it as a dictionary
    :return:
    """
    with open(train_file, 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(val_file, 'rb') as f:
        X_val, y_val = pickle.load(f)
    with open(test_file, 'rb') as f:
        X_test, y_test = pickle.load(f)
    data = {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test}
    return data


def load_raw_images():
    directory = 'Damage_Classification_Extended_V2'
    train_X, train_y = load_damage_classification_imgs(
        minor_dir='../Data/Images/' + directory + '/training/01-minor',
        moderate_dir='../Data/Images/' + directory + '/training/02-moderate',
        major_dir='../Data/Images/' + directory + '/training/03-severe', small_images=True)
    train_X = np.reshape(train_X.numpy(), (train_X.shape[0], -1))
    train_y = train_y.numpy()
    val_X, val_y = load_damage_classification_imgs(
        minor_dir='../Data/Images/' + directory + '/validation/01-minor',
        moderate_dir='../Data/Images/' + directory + '/validation/02-moderate',
        major_dir='../Data/Images/' + directory + '/validation/03-severe', small_images=True)
    val_X = np.reshape(val_X.numpy(), (val_X.shape[0], -1))
    val_y = val_y.numpy()
    test_X, test_y = load_damage_classification_imgs(
        minor_dir='../Data/Images/' + directory + '/test/01-minor',
        moderate_dir='../Data/Images/' + directory + '/test/02-moderate',
        major_dir='../Data/Images/' + directory + '/test/03-severe', small_images=True)
    test_X = np.reshape(test_X.numpy(), (test_X.shape[0], -1))
    test_y = test_y.numpy()
    data = {'X_train': train_X, 'y_train': train_y, 'X_val': val_X, 'y_val': val_y, 'X_test': test_X, 'y_test': test_y}
    return data


def load_images_ImageFolder(train_dir, val_dir, test_dir):
    train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    val_test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    train_dataset = ImageFolder(train_dir, train_transforms)
    val_dataset = ImageFolder(val_dir, val_test_transforms)
    test_dataset = ImageFolder(test_dir, val_test_transforms)
    print test_dataset.imgs
    return train_dataset, val_dataset, test_dataset


# train, val, test = load_images_ImageFolder('../OriginalData/car-damage-dataset/data3a/training',
#                         '../OriginalData/car-damage-dataset/data3a/validation',
#                         '../OriginalData/car-damage-dataset/data3a/validation')


#
# get_img_features_split('./Data/vgg16_extracted_features.pkl')


# inp_dirs = ['./Original_Data/car-damage-dataset/data1a/training/00-damage/',
#             './Original_Data/car-damage-dataset/data1a/validation/00-damage/',
#             './Original_Data/DCD-master/DCD-1/01/',
#             './Original_Data/DCD-master/DCD-1/02/',
#             './Original_Data/DCD-master/DCD-1/03/']
# save_damaged_car_images(inp_dirs, out_dir)
# inp_dir = './car_ims/'
# out_dir = './Data/Images/Stanford_Car_Dataset/'
# resize_stanford_dataset(inp_dir, out_dir)

# load_stanford_annotations('./Stanford_Dataset/cars_annos.mat', './Data/original_stanford_metadata.pkl')
# extract_good_images(1825, './Data/original_stanford_metadata.pkl', './Data/Images/Original_Stanford_Data/',
#                     './Data/good_images_metadata.pkl', './Data/Images/Good/',
#                     './Data/pretraining_stanford_metadata.pkl', './Data/Images/Pretraining_Stanford_Data/')

