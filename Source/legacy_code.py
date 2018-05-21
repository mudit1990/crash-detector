
# def check_accuracy(self, X_train, y_train, X_val, y_val):
#     """
#     LEGACY CODE FOR CHECK ACCURACIES FROM VGG_NET.PY
#     Check the accuracy given by the model for X_train, y_train, X_val, y_val for the given model
#     :param X_train: Tensor
#     :param y_train: Tesnor
#     :param X_val: Tensor
#     :param y_val: Tensor
#     :return:
#     """
#     # set the model to eval mode
#     start = time.time()
#     self.model.eval()
#     # choose a subset of training and validation data
#     rd_idx = np.random.choice(range(len(X_val)), 100)
#     X_train, y_train = X_train[rd_idx], y_train[rd_idx]
#     X_val, y_val = X_val[rd_idx], y_val[rd_idx]
#     print X_train.shape, X_val.shape
#     val_scores = self.model.forward(Variable(X_val, requires_grad=False)).data.numpy()
#     y_val_pred = np.argmax(val_scores, axis=1)
#     val_acc = np.mean(y_val.numpy() == y_val_pred)
#     train_scores = self.model.forward(Variable(X_train, requires_grad=False)).data.numpy()
#     y_train_pred = np.argmax(train_scores, axis=1)
#     train_acc = np.mean(y_train.numpy() == y_train_pred)
#     end = time.time()
#     print 'Accuracy checks done in ', (end - start)
#     return train_acc, val_acc


# def legacy_main_vgg_net():
#     X_train, y_train = data_utils.load_damage_classification_imgs(
#     minor_dir='../OriginalData/car-damage-dataset/data3a/training/01-minor',
#     moderate_dir='../OriginalData/car-damage-dataset/data3a/training/02-moderate',
#     major_dir='../OriginalData/car-damage-dataset/data3a/training/03-severe')
#     print 'training data shape', X_train.shape, y_train.shape
#     X_val, y_val = data_utils.load_damage_classification_imgs(
#     minor_dir='../OriginalData/car-damage-dataset/data3a/validation/01-minor',
#     moderate_dir='../OriginalData/car-damage-dataset/data3a/validation/02-moderate',
#     major_dir='../OriginalData/car-damage-dataset/data3a/validation/03-severe')
#     print 'validation data shape', X_val.shape, y_val.shape
#     # creating a tensor dataset to use in dataloader
#     # dataset = TensorDataset(X_train, y_train)

# def legacy_vgg_net_cuda_main():
#     X_train, y_train = data_utils.load_damage_classification_imgs(
#         minor_dir='../OriginalData/car-damage-dataset/data3a/training/01-minor',
#         moderate_dir='../OriginalData/car-damage-dataset/data3a/training/02-moderate',
#         major_dir='../OriginalData/car-damage-dataset/data3a/training/03-severe')
#     print 'training data shape', X_train.shape, y_train.shape
#     X_val, y_val = data_utils.load_damage_classification_imgs(
#         minor_dir='../OriginalData/car-damage-dataset/data3a/validation/01-minor',
#         moderate_dir='../OriginalData/car-damage-dataset/data3a/validation/02-moderate',
#         major_dir='../OriginalData/car-damage-dataset/data3a/validation/03-severe')
#     print 'validation data shape', X_val.shape, y_val.shape
#     # creating a tensor dataset to use in dataloader
#     dataset = TensorDataset(X_train, y_train)


# def check_accuracy_cuda_code(self, X_train, y_train, X_val, y_val):
#     """
#     Check the accuracy given by the model for X_train, y_train, X_val, y_val for the given model
#     :param X_train: Tensor
#     :param y_train: Tesnor
#     :param X_val: Tensor
#     :param y_val: Tensor
#     :return:
#     """
#     # set the model to eval mode
#     start = time.time()
#     self.model.eval()
#     # choose a subset of training and validation data
#     rd_idx = np.random.choice(range(len(X_val)), 100)
#     X_train, y_train = X_train[rd_idx], y_train[rd_idx]
#     X_val, y_val = X_val[rd_idx], y_val[rd_idx]
#     print X_train.shape, X_val.shape
#     val_scores = self.model.forward(Variable(X_val.cuda(), requires_grad=False)).cpu().data.numpy()
#     y_val_pred = np.argmax(val_scores, axis=1)
#     val_acc = np.mean(y_val.numpy() == y_val_pred)
#     train_scores = self.model.forward(Variable(X_train.cuda(), requires_grad=False)).cpu().data.numpy()
#     y_train_pred = np.argmax(train_scores, axis=1)
#     train_acc = np.mean(y_train.numpy() == y_train_pred)
#     end = time.time()
#     print 'Accuracy checks done in ', (end - start)
#     return train_acc, val_acc


# def extract_good_images(num_images, src_meta_file, src_dir, good_meta_file, good_dir, stanford_meta_file, stanford_dir):
#     """
#     Randomly extracts num_images from stanford dataset to get images of undamaged cars. These images are then
#     removed from the stanford dataset. The meta files are also split into two parts, one for good images extracted
#     and one for the remaining stanford dataset images that will be used for pre-training later
#     """
#     # open the stanford meta-data pickle file
#     with open(src_meta_file, 'rb') as f:
#         meta_data = pickle.load(f)
#     # creating a set of good_idx for faster lookup
#     good_idx = set(np.random.choice(len(meta_data), num_images, replace=False))
#     stanford_data = []
#     good_data = []
#     print_after_iter = 1000
#     for i in range(len(meta_data)):
#         if i%print_after_iter == 0:
#             print i, 'files done!'
#         file_name = meta_data[i][0]
#         src_path = os.path.join(src_dir, file_name)
#         if i in good_idx:
#             good_data.append(meta_data[i])
#             # copy the images to new folder
#             shutil.copy(src_path, good_dir)
#         else:
#             stanford_data.append(meta_data[i])
#             # copy the images to new folder
#             shutil.copy(src_path, stanford_dir)
#     with open(stanford_meta_file, 'wb') as f:
#         pickle.dump(stanford_data, f)
#     with open(good_meta_file, 'wb') as f:
#         pickle.dump(good_data, f)
