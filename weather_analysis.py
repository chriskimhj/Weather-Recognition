import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

TEST_PATH = "D:/hjchr/School/5th Year/ML/Project/Final/personal_archives"
DATA_PATH = "D:/hjchr/School/5th Year/ML/Project/Final/data"
CLASS_NAMES = ["cloudy", "rain", "shine", "sunrise"]

def collect_images(path):
    """
    Read images from files and extract label from filename
    :param path: String holding the absolute path to the training dataset
    :return: imgs:      List of image data
             labels:    List of labels
             filenames: List of file names
    """
    imgs = []
    labels = []
    filenames = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = img[:, :, ::-1]
        imgs.append(img)
        filenames.append(filename)
        for i in range(len(CLASS_NAMES)):
            if CLASS_NAMES[i] in filename:
                labels.append(i)
    print(f"All images collected. ({len(imgs)})")
    return imgs, labels, filenames

def preprocess_images(imgs):
    """
    Take list of images and preprocess them by:
        1. Resize: To fit a 300 by 300 shape
    Note: Grayscale not used as colors are very important traits for weather recognition.
    :param imgs: List of original images
    :return: List of preprocessed images
    """
    res = []
    for img in imgs:
        re_img = cv2.resize(img, (300, 300))
        res.append(re_img)
    print(f"All images preprocessed.({len(res)})")
    return res

def split_dataset(dataset, labels, files):
    """
    Splits the dataset into training set and testing sets.
    The same applies for class labels and filenames.
    :param dataset: List of image data
    :param labels: List of image labels
    :param files: List of image filenames
    :return: Separate lists for train and test lists for each input
    """
    # Split to train test (80%, 20%)
    data_count = dataset.shape[0]
    train_frac = 0.8
    train_count = int(data_count * train_frac)
    permuted_idx = np.random.permutation(dataset.shape[0])

    train_imgs = dataset[permuted_idx[0:train_count]]
    test_imgs = dataset[permuted_idx[train_count+1:]]
    train_labels = labels[permuted_idx[0:train_count]]
    test_labels = labels[permuted_idx[train_count+1:]]
    train_files = files[permuted_idx[0:train_count]]
    test_files = files[permuted_idx[train_count+1:]]

    print(f"Training Set: {train_imgs.shape[0]}")
    print(f"Testing Set: {test_imgs.shape[0]}")
    return train_imgs, test_imgs, train_labels, test_labels, train_files, test_files

def validate_labels(train_labels, train_files, test_labels, test_files):
    """
    Checks if each file was correctly labelled by referring to original file name.
    :param train_labels: List of labels for training set
    :param train_files:  List of filenames for training set
    :param test_labels:  List of labels for testing set
    :param test_files:   List of filenames for testing set
    :return: List of incorrectly labelled files.
    """
    error_files = []

    for i in range(train_labels.shape[0]):
        if CLASS_NAMES[train_labels[i]] not in train_files[i]:
            print(f"INCORRECT LABEL ON FILE: {train_files[i]}")
            error_files.append(train_files[i])

    for i in range(test_labels.shape[0]):
        if CLASS_NAMES[test_labels[i]] not in test_files[i]:
            print(f"INCORRECT LABEL ON FILE: {test_files[i]}")
            error_files.append(test_files[i])

    return error_files

def show_sample_images(imgs, labels, title, count):
    """
    Plots a 5 by 5 grid of images from the input list with their class names
    :param imgs:    List of image data
    :param labels:  List of image labels
    :return:    null
    """
    plt.figure(figsize=(10, 10))
    for i in range(count):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i])
        plt.xlabel(CLASS_NAMES[labels[i]])
    plt.suptitle(title)
    plt.show()

def build_model():
    """
    Build the CNN model referring to tensorflow tutorial
    :return:    The built CNN model
    """
    # Sequential Structure
    model = models.Sequential()

    # Adding a convolutional layer with size 32 kernel and 3 channels (RGB)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
    # Pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Adding another convolutional layer with size 64 kernel and 3 channels (RGB)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # Second pooling layer
    model.add(layers.MaxPooling2D((2, 2)))
    # Adding a final convolutional layer with size 64 kernel and 3 channels (RGB)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # Flattening 3D tensor into 1D vector
    model.add(layers.Flatten())
    # Creating output layer
    model.add(layers.Dense(64, activation='relu'))
    # Output layer must be able to classify 4 classes
    model.add(layers.Dense(4))
    # Compiling the built model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def plot_accuracy(history):
    """
    Display the accuracy of learning of the built CNN model
    :param history: Object containing learning results
    :return: null
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def plot_loss(history):
    """
    Display the loss of learning of the built CNN model
    :param history:  Object containing learning results
    :return: null
    """
    # Plot loss
    plt.plot(history.history['loss'], label="loss")
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def compute_confusion_matrix(true, pred):
    K = len(np.unique(true)) # Number of classes
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    for i in range(len(CLASS_NAMES)):
        print(f"\nImage Class: {CLASS_NAMES[i]}")
        print(f"{result[i][0]} images classified as {CLASS_NAMES[0]}")
        print(f"{result[i][1]} images classified as {CLASS_NAMES[1]}")
        print(f"{result[i][2]} images classified as {CLASS_NAMES[2]}")
        print(f"{result[i][3]} images classified as {CLASS_NAMES[3]}")
    print("\nConfusion Matrix")
    print(result)
    return result

def main():
    # Build CNN Model
    model = build_model()
    model.summary()

    # Train model using training dataset
    print("\nTraining Model...")
    train(model)

    # Test model with 10 new images
    print("\nTesting Model...")
    test(model)

def train(model):

    # Acquire the data from local path
    og_imgs, labels, filenames = collect_images(DATA_PATH)

    # Preprocess the data and convert to numpy
    imgs = preprocess_images(og_imgs)
    imgs = np.array(imgs)
    labels = np.array(labels)
    filenames = np.array(filenames)

    # Split dataset to train and test sets
    train_imgs, test_imgs, train_labels, test_labels, train_files, test_files = split_dataset(imgs, labels, filenames)
    train_imgs, test_imgs = train_imgs/255.0, test_imgs/255.0

    # Validate labels
    error_files = validate_labels(train_labels, train_files, test_labels, test_files)
    if len(error_files) > 0:
        print("Files labelled incorrectly.")

    # Plot a few sample images from each set
    show_sample_images(train_imgs, train_labels, "Training Set", 25)
    show_sample_images(test_imgs, test_labels, "Testing Set", 25)

    # Apply Training
    history = model.fit(train_imgs, train_labels, epochs=10,
                        validation_data=(test_imgs, test_labels))

    #################################
    #   Testing on 20% Testing Set  #
    #################################

    # Trained model finished, predict
    pred_labels = model.predict(test_imgs)
    pred_labels_t = []
    for i in range(len(pred_labels)):
        pred_labels_t.append(np.argmax(pred_labels[i]))
    pred_labels_t = np.array((pred_labels_t))
    pred_labels_t = pred_labels_t.astype(int)

    # Compute confusion matrix
    confusion_mx = compute_confusion_matrix(test_labels, pred_labels_t)

    # Show prediction result
    show_sample_images(test_imgs, pred_labels_t, "Predicted Set", 25)

    # Display Accuracy Plot per Epoch
    plot_accuracy(history)

    # Plotting loss
    plot_loss(history)

    # Calculate and display result test loss and test accuracy
    test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
    print("\nTest Accuracy:\t" + str(test_acc))
    print("Test Loss:\t" + str(test_loss))

def test(model):

    ################################
    #   Testing on 10 new images   #
    ################################

    # Acquire the data from local path
    og_imgs, labels, filenames = collect_images(TEST_PATH)

    # Preprocess the data and convert to numpy
    imgs = preprocess_images(og_imgs)
    imgs = np.array(imgs)
    labels = np.array(labels)
    filenames = np.array(filenames)

    # Plot a few sample images from each set
    show_sample_images(imgs, labels, "New Test Set", 10)

    # Trained model finished, predict
    pred_labels = model.predict(imgs)
    pred_labels_t = []
    for i in range(len(pred_labels)):
        pred_labels_t.append(np.argmax(pred_labels[i]))
    pred_labels_t = np.array((pred_labels_t))
    pred_labels_t = pred_labels_t.astype(int)

    # Compute confusion matrix
    confusion_mx = compute_confusion_matrix(labels, pred_labels_t)

    # Show prediction result
    show_sample_images(imgs, pred_labels_t, "Predicted Set", 10)

    # Calculate and display result test loss and test accuracy
    test_loss, test_acc = model.evaluate(imgs, labels, verbose=2)
    print("\nTest Accuracy:\t" + str(test_acc))
    print("Test Loss:\t" + str(test_loss))

if __name__=="__main__":

    main()