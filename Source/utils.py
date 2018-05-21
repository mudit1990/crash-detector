import numpy as np
from PIL import Image
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from sklearn.model_selection import train_test_split
import shutil
import pickle


def is_greyscale(img):
    sh = np.array(img).shape
    if len(sh) < 3:
        return True
    else:
        return False


def count_greyscale():
    cnt = 0
    inp_dir = './Data/Images/Original_Stanford_Data/'
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    for f in files:
        if f == '.DS_Store':
            continue
        img_path = os.path.join(inp_dir, f)
        sh = np.array(Image.open(img_path)).shape
        if len(sh) < 3:
            cnt += 1
    print cnt


def assign_labels_colors(labels, colors):
    """
    Takes a list of labels and colors and assigns a unique label to each color. Returns a color_list of length(labels).
    The colors will loop around if the number of unique labels are more than the number of unique colors
    :param labels:
    :param colors:
    :return: color_list
    """
    col_idx = 0
    label2col = {}
    col_list = []
    for i in range(len(labels)):
        if labels[i] in label2col:
            col_list.append(label2col[labels[i]])
        else:
            col = colors[col_idx % len(colors)]
            col_idx += 1
            label2col[labels[i]] = col
            col_list.append(col)
    return col_list


def plot_tsne(X, y, title):
    colors = ['#ff0000', '#00ff00', '#0000ff']
    col_list = assign_labels_colors(y, colors)

    tsne = TSNE(verbose=1, method='exact')
    start = time.time()
    tsne_dims = tsne.fit_transform(X)
    end = time.time()
    print('time taken by TSNE ', (end - start))
    plt.scatter(tsne_dims[:, 0], tsne_dims[:, 1], c=col_list, s=75, alpha=0.25)
    plt.tight_layout()
    plt.title(title)
    outfile = '../plots/' + title + '.png'
    plt.savefig(outfile)
    # plt.show()


def plot_weights_heapmap(wt, size):
    """
    Takes a 1-d weight vector and converts it into a sqaure heatmap
    :return:
    """
    wt_sqr = wt.reshape(size, size)
    plt.imshow(wt_sqr, cmap='hot')
    plt.show()


def check_duplicate_names(dir):
    subdirs = ['01-minor', '02-moderate', '03-severe']
    all_files = []
    for subdir in subdirs:
        dir_path = os.path.join(dir, subdir)
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        all_files += files
    print len(all_files)
    print np.unique(all_files).shape


def get_filename2label(dir):
    filename2label = {}
    subdirs = ['01-minor', '02-moderate', '03-severe']
    for subdir in subdirs:
        dir_path = os.path.join(dir, subdir)
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for fl in files:
            filename2label[fl] = subdir
    return filename2label


def compare_labelling(labels1, labels2):
    mismatch_count = 0
    for lbl in labels1.keys():
        val1 = labels1.get(lbl, '')
        val2 = labels2.get(lbl, '')
        if val1 != val2:
            mismatch_count += 1
            print 'Mudit labelled:', lbl, val1
            print 'Mohit labelled:', lbl, val2
    print 'Total mismatch count:', mismatch_count


def check_datasets_size(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    for data in data_loader:
        X, y = data
    return X.shape, y.shape


def split_images_train_val(inp_dir, val_dir, val_ratio):
    """
    take all images from inp_dir, split them on the bases of val_ratio and move the validation images to folder
    specified by val_dir
    """
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    train_files, val_files = train_test_split(files, test_size=val_ratio, random_state=42, shuffle=True)
    move_files(inp_dir, val_files, val_dir)


def split_images_train_val_test(inp_dir, train_dir, val_dir, test_dir, test_ratio, val_ratio):
    """
    takes all images from inp_dir and splits them into train, validation and test on the basis of val_ratio and test
    ratio 
    """
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=192690, shuffle=True)
    train_files, val_files = train_test_split(train_files, test_size=val_ratio, random_state=192690, shuffle=True)
    print len(train_files)
    print len(val_files)
    print len(test_files)
    move_files(inp_dir, train_files, train_dir, leave_copy=True)
    move_files(inp_dir, val_files, val_dir, leave_copy=True)
    move_files(inp_dir, test_files, test_dir, leave_copy=True)


def move_files(inp_dir, files, out_dir, leave_copy=False):
    """
    Moves all the files specified in files from inp_dir to out_dir. This is meant as a helper function to
    split_images_train_val and split_images_train_val_test
    """
    for f in files:
        if leave_copy:
            shutil.copy(os.path.join(inp_dir, f), os.path.join(out_dir, f))
        else:
            shutil.move(os.path.join(inp_dir, f), os.path.join(out_dir, f))


def collect_images(inp_dirs, out_dir):
    """
    reads all the image files present in all the inp_dirs and saves them in out_dir by
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
            shutil.move(src, dst)
            if file_num % print_after_iter == 0:
                print file_num, 'files processed!'


def resize_stanford_dataset(inp_dir, out_dir):
    """
    The function resizes all the images in stanford dataset to 224x224
    """
    print_after_iter = 1000
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    for i in range(len(files)):
        if i % print_after_iter == 0:
            print i, 'files resized!'
        src = os.path.join(inp_dir, files[i])
        dst = os.path.join(out_dir, files[i])
        img = Image.open(src).resize((224, 224))
        img.save(dst)


def split_stanford_data_ImageFolder(meta_file, inp_dir, train_dir, val_dir, test_dir, val_ratio, test_ratio):
    """
    Splits the stanford data into train, test, val and then re-arranges them into folders based on the class it belongs
    to as required by ImageFolder
    """
    # load the stanford annotations
    with open(meta_file, 'rb') as f:
        meta_data = pickle.load(f)
    image2label = {r[0]: r[1] for r in meta_data}
    files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f))]
    train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42, shuffle=True)
    train_files, val_files = train_test_split(train_files, test_size=val_ratio, random_state=42, shuffle=True)
    copy_stanford_files_ImageFolder(train_files, inp_dir, train_dir, image2label)
    print 'Copied training images'
    copy_stanford_files_ImageFolder(val_files, inp_dir, val_dir, image2label)
    print 'Copied validation images'
    copy_stanford_files_ImageFolder(test_files, inp_dir, test_dir, image2label)
    print 'Copied test images'


def copy_stanford_files_ImageFolder(files, inp_dir, out_dir, image2label):
    for f in files:
        lb = image2label[f]
        lb_dir = os.path.join(out_dir, str(lb))
        if not os.path.exists(lb_dir):
            os.mkdir(lb_dir)
        shutil.copy(os.path.join(inp_dir, f), os.path.join(lb_dir, f))


def plot_classification_images(img_dir, num_images=3):
    """
    plots num_images for each class (after resizing to 224x224) from img_path
    """
    dirs = sorted(os.listdir(img_dir))
    dirs.remove('.DS_Store')  # remove mac ds_store
    num_classes = len(dirs)
    plt.figure(figsize=(10, 10))
    for y, y_dir in enumerate(dirs):
        dir_path = os.path.join(img_dir, y_dir)
        chosen_files = __get_random_files(num_images, dir_path, y_dir)
        chosen_imgs = __get_resized_images(dir_path, chosen_files)
        for i in range(num_images):
            plt_idx = i * num_images + y + 1
            plt.subplot(num_images, num_classes, plt_idx)
            plt.imshow(np.asarray(chosen_imgs[i]))
            plt.axis('off')
            if i == 0:
                plt.title(__get_classname(y_dir), fontsize=20)
    plt.tight_layout()
    outfile = '../plots/' + 'damage_cars_classification' + '.png'
    plt.savefig(outfile)
    # plt.show()


def __get_random_files(num_images, dir_path, ydir):
    if ydir == '01-minor':
        chosen_files = ['image_0003.JPEG', 'image_0007.jpeg', 'image_0012.JPEG']
    elif ydir == '02-moderate':
        chosen_files = ['image_0002.JPEG', 'image_0007.JPEG', 'image_0075.JPEG']
    else:
        chosen_files = ['image_0081.JPEG', 'image_0023.JPEG', 'image_0048.JPEG']
    # files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    # chosen_idx = np.random.choice(len(files), num_images)
    # chosen_files = [files[i] for i in chosen_idx]
    return chosen_files


def __get_classname(dir):
    if dir == '01-minor': return 'Minor'
    if dir == '02-moderate':
        return 'Moderate'
    else:
        return 'Severe'


def __get_resized_images(dir_path, files):
    imgs = []
    for f in files:
        img_path = os.path.join(dir_path, f)
        img = Image.open(img_path).resize((224, 224))
        imgs.append(img)
    return imgs


def plot_bar_charts(labels, vals, ylabel, title, ymax, baseline=None):
    # plt.figure(figsize=(5, 5))
    # plt.rc('ytick', labelsize=10)
    x_pos = np.arange(len(labels))
    if baseline is not None:
        line_x = np.arange(-1, len(labels) + 1)
        baseline_y = np.zeros(len(line_x)) + baseline
        line, = plt.plot(line_x, baseline_y, color='k', label='Baseline', linewidth=2)
        plt.legend()
    plt.bar(x_pos, vals, align='center', width=0.3, color=['#EC7063', '#F4D03F', '#3498DB']) #color='#3498DB')
    plt.xticks(x_pos, labels)
    plt.ylabel(ylabel)
    plt.ylim((0, ymax))
    # plt.title('Dominance')
    plt.title(title)
    plt.tight_layout()
    outfile = '../plots/' + title + '.png'
    # plt.show()
    plt.savefig(outfile)


def plot_group_bar_charts_Acc(labels, vals, ylabel, xlabel, title, ymax, baseline=None, display_plot=False):
    cols = vals.shape[0]
    N = 2 * vals.shape[1] + 1

    colours = ['#5DADE2', '#F4D03F', '#AAB7B8']
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()

    legend_colours = []
    for col in xrange(cols):
        this_vals = np.zeros(N)
        this_vals[1:N:2] = vals[col, :]
        this_rects = ax.bar(ind + col * width, this_vals, width, color=colours[col % len(colours)])
        legend_colours.append(this_rects[0])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xlabel)

    if baseline is not None:
        ax.plot(ind, np.zeros(len(ind))+baseline, c='k', linewidth=2)

    ax.legend(legend_colours, labels, loc="upper left")

    plt.ylim((0, ymax))
    plt.tight_layout()

    if display_plot:
        plt.show()
    else:

        outfile = '../Plots/' + title + '.png'
        plt.savefig(outfile)


def plot_group_bar_charts_Fscore(labels, vals, ylabel, xlabel, title, ymax, display_plot=False):
    cols = vals.shape[0]
    N = 2 * vals.shape[1] + 1

    colours = ['#EC7063', '#2ECC71', '#AF7AC5']
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()

    legend_colours = []
    for col in xrange(cols):
        this_vals = np.zeros(N)
        this_vals[1:N:2] = vals[col, :]
        color = np.array(['#000000'] * N)
        color[1:N:2] = np.array(colours)
        # color = colours[col % len(colours)]
        this_rects = ax.bar(ind + col * width, this_vals, width, color=color)
    legend_colours = this_rects[1:N:2]

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xlabel)

    ax.legend(legend_colours, labels, loc="upper left")

    plt.ylim((0, ymax))
    plt.tight_layout()

    if display_plot:
        plt.show()
    else:

        outfile = '../Plots/' + title + '.png'
        plt.savefig(outfile)


def plot_finetuning_effect(x, y1, y2, label1, label2, title, xlabel, ylabel):
    plt.plot(x, y1, label=label1, linewidth=2)
    plt.plot(x, y2, label=label2, linewidth=2)
    plt.legend(loc=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    outfile = '../Plots/' + title + '.png'
    plt.savefig(outfile)

# x = np.arange(16)
# y1 = [0.69, 15.21, 41.69, 55.21, 67.10, 72.89, 71.96, 74.13, 74.44, 73.12, 71.58, 77.45, 77.91, 74.67, 77.76, 77.22] # 1e-4
# y2 = [0.77, 02.39, 14.05, 30.65, 39.38, 45.17, 51.66, 55.67, 59.84, 62.00, 64.09, 65.32, 68.33, 68.57, 69.03, 69.49] # 1e-5
# plot_finetuning_effect(x, y1, y2, 'lr: 1e-4', 'lr: 1e-5', 'Stanford Validation Accuracy', 'No of Epochs', 'Accuracy %')

# x = [3,5,7,9,11,13]
# y = [78.37,78.68,74.74,72.66,65.79,52.2]
# plt.plot(x, y, linewidth=2)
# plt.title('Stanford Validation Accuracy vs Starting fine-tuning layer')
# plt.xlabel('Layer from where fine-tuning starts')
# plt.ylabel('Validation Accuracy')
# plt.tight_layout()
# outfile = '../Plots/' + 'Stan_fine_tunevs_layers' + '.png'
# plt.savefig(outfile)

# x = [5, 10, 15]
# y1 = [65.9, 68.04, 62.88] # 1e-4
# y2 = [68.58, 70.10, 76.78] # 1e-5
# plot_finetuning_effect(x, y1, y2, 'first-stage lr: 1e-4', 'first-stage lr: 1e-5', 'Damage Classification Validation Accuracy', 'No of Epochs (of Stanford Pretraining)', 'Accuracy %')


# labels = ['Validation', 'Test']
# title = 'Validation & Test Accuracy from Models'
# vals = np.array([[46.43, 66.96, 78.57, 76.78],
#                  [40.72, 62.37, 70.1, 74.22]])
# ylabel = 'Accuracy'
# xlabel = ('', 'svc_image_pixels', '', 'svc_vgg_feats', '', 'single-fine-tune', '', 'multi-fine-tune')
# plot_group_bar_charts_Acc(labels, vals, ylabel, xlabel, title, ymax=100, baseline=71.0, display_plot=False)

# labels = ['Minor', 'Moderate', 'Severe']
# title = 'F-score per class (On test set)'
# vals = np.array([[0.68, 0.78],
#                  [0.61, 0.64],
#                  [0.81, 0.80]])
# ylabel = 'F-score'
# xlabel = ('', 'single-fine-tune', '', 'multi-fine-tune')
# plot_group_bar_charts_Acc(labels, vals, ylabel, xlabel, title, ymax=1.0, display_plot=False)

# labels = ['Minor','Moderate','Severe']
# vals = [31.91, 32.92, 35.16]
# ylabel = 'Distribution Percentage'
# title = 'Damaged Cars Class Distribution'
# plot_bar_charts(labels, vals, ylabel, title, ymax=50.0, baseline=None)

# labels = ['','Constant LR','Differential LR','']
# vals = [0, 76.78, 78.57, 0]
# ylabel = 'Accuracy %'
# title = 'Validation Accuracy on Damage Classification'
# plot_bar_charts(labels, vals, ylabel, title, ymax=100.0, baseline=None)


# plot_classification_images('../Data/Images/Damage_Classification_Extended_V2/data')

# resize_stanford_dataset('../Data/Images/Stanford_Dataset/car_ims/',
#  '../Data/Images/Stanford_Dataset/car_ims_resized/')

# split_stanford_data_ImageFolder('../Data/Objects/original_stanford_metadata.pkl',
#                                 '../Data/Images/Stanford_Dataset/Images',
#                                 '../Data/Images/Stanford_Dataset/training',
#                                 '../Data/Images/Stanford_Dataset/validation',
#                                 '../Data/Images/Stanford_Dataset/test',0.1,0.2)

# inp_dirs = ['../Data/Images/Damage_Classification_Extended_V2/data1/01-minor',
#             '../Data/Images/Damage_Classification_Extended_V2/data2/01-minor',
#             '../Data/Images/Damage_Classification_Extended_V2/data3/01-minor']
# collect_images(inp_dirs, '../Data/Images/Damage_Classification_Extended_V2/data/01-minor')

# split_images_train_val('../Data/Images/Damage_Classification/training/03-severe/',
#                        '../Data/Images/Damage_Classification/validation/03-severe/', 0.1)

# split_images_train_val_test('../Data/Images/Damage_Classification_Extended_V2/data/01-minor',
#                             '../Data/Images/Damage_Classification_Extended_V2/training/01-minor',
#                             '../Data/Images/Damage_Classification_Extended_V2/validation/01-minor',
#                             '../Data/Images/Damage_Classification_Extended_V2/test/01-minor',
#                             0.15, 0.10)

# mudit_labels = get_filename2label('../OriginalData/DCD-master/Mudit')
# mohit_labels = get_filename2label('../OriginalData/DCD-master/Mohit')
# compare_labelling(mudit_labels, mohit_labels)
