import numpy
import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

IMAGE_DIMENSION = 197
IMAGE_SHAPE = (IMAGE_DIMENSION, IMAGE_DIMENSION)


class PreTrainedModel(object):
    def __init__(self, model_name, base_model):
        self.model_name = model_name
        self.base_model = base_model

    def prepare_model(self):
        # add a global spatial average pooling layer
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)

        # and a logistic layer
        predictions = Dense(2, activation='softmax')(x)

        # this is the model we will train
        return Model(inputs=self.base_model.input, outputs=predictions)

    def train_only_top_layers(self, model, train_tensors, train_targets, valid_tensors, valid_targets):
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # train the model on the new data for a few epochs
        model.fit(train_tensors, train_targets, batch_size=16, epochs=1, verbose=2,
                  validation_data=(valid_tensors, valid_targets))

    def train_remaining_top_layers(self, model, train_tensors, train_targets, valid_tensors, valid_targets):
        number_of_layers_to_freeze = int(len(model.layers) * 0.75)

        # we will freeze the first three fourth of layers and unfreeze the rest:
        for layer in model.layers[:number_of_layers_to_freeze]:
            layer.trainable = False

        for layer in model.layers[number_of_layers_to_freeze:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        checkpointer = ModelCheckpoint(
            filepath=self.model_name + '.weights.best.from_scratch.hdf5',
            verbose=1,
            save_best_only=True)

        # we train our model again (this time fine-tuning the top 2 inception blocks alongside the top Dense layers)
        model.fit(train_tensors, train_targets, batch_size=16, epochs=3, verbose=2,
                  callbacks=[checkpointer], validation_data=(valid_tensors, valid_targets))

    def train_and_evaluate(self, train_tensors, train_targets, valid_tensors, valid_targets, test_tensors,
                           test_targets):

        print("\n\nStarting for model: " + self.model_name + " ...\n")

        model = self.prepare_model()

        start_time = time.time()

        # first: train only the top layers (which were randomly initialized)
        self.train_only_top_layers(model, train_tensors, train_targets, valid_tensors, valid_targets)

        # at this point, the top layers are well trained and we can start fine-tuning convolutional layers. We will
        # freeze the bottom N layers and train the remaining top layers.
        self.train_remaining_top_layers(model, train_tensors, train_targets, valid_tensors, valid_targets)

        print("\n--- %s seconds ---" % (time.time() - start_time))

        # evaluate
        test(model, test_tensors, test_targets)


def load_dataset(path):
    data = load_files(path)
    files = numpy.array(data['filenames'])
    targets = np_utils.to_categorical(numpy.array(data['target']), len(data['target_names']))

    return files, targets


def print_dataset_statistics(train_files, valid_files, test_files):
    print('There are %s total images.' % len(numpy.hstack([train_files, valid_files, test_files])))
    print('There are %d training images.' % len(train_files))
    print('There are %d validation images.' % len(valid_files))
    print('There are %d test images.\n\n' % len(test_files))


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=IMAGE_SHAPE)

    # convert PIL.Image.Image type to 3D tensor
    three_d_tensor = image.img_to_array(img)

    # convert 3D tensor to 4D tensor and return 4D tensor
    return numpy.expand_dims(three_d_tensor, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return numpy.vstack(list_of_tensors)


def file_to_tensor(files):
    return paths_to_tensor(files).astype('float32') / 255


def prepare_pre_trained_models():
    inception_base_model = InceptionV3(include_top=False, weights='imagenet',
                                       input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

    res_net_50_base_model = ResNet50(include_top=False, weights='imagenet',
                                     input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

    xception_base_model = Xception(include_top=False, weights='imagenet',
                                   input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

    vgg_19_base_model = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

    inception_v3 = PreTrainedModel('InceptionV3', inception_base_model)
    resnet50 = PreTrainedModel('ResNet50', res_net_50_base_model)
    xception = PreTrainedModel('Xception', xception_base_model)
    vgg19 = PreTrainedModel('VGG19', vgg_19_base_model)

    return [inception_v3, resnet50, xception, vgg19]


def prepare_own_model():
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                     input_shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3)))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=1024, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation='softmax'))

    return model


def test(model, test_tensors, test_targets):
    # get index of predicted category for each image in test set
    predictions = [numpy.argmax(model.predict(numpy.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report scores
    y_test = numpy.argmax(test_targets, axis=1)

    print('\nAccuracy score: ', format(accuracy_score(y_test, predictions)))
    print('Precision score: ', format(precision_score(y_test, predictions)))
    print('Recall score: ', format(recall_score(y_test, predictions)))
    print('F1 score: ', format(f1_score(y_test, predictions, average='micro')))


print("\nLoading training, validation and test datasets ...\n")

# load train, validation and test datasets
train_files, train_targets = load_dataset('train')
valid_files, valid_targets = load_dataset('valid')
test_files, test_targets = load_dataset('test')

# print statistics about the dataset
print_dataset_statistics(train_files, valid_files, test_files)

print("Converting files to tensors ...\n")

# pre-process the data for Keras
train_tensors = file_to_tensor(train_files)
valid_tensors = file_to_tensor(valid_files)
test_tensors = file_to_tensor(test_files)

print("\nFinished loading dataset and converting files to tensors.\n\n")

# Train and evaluate own model
print("\nStarting for own model ...\n")

model = prepare_own_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(train_tensors, train_targets, batch_size=16, epochs=3, verbose=2,
          callbacks=[
              ModelCheckpoint(filepath='own_model.weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
          ],
          validation_data=(valid_tensors, valid_targets))

test(model, test_tensors, test_targets)

# Train and evaluate pre-trained models(InceptionV3, ResNet50, Xception and VGG19)
for pre_trained_model in prepare_pre_trained_models():
    pre_trained_model.train_and_evaluate(train_tensors, train_targets, valid_tensors, valid_targets, test_tensors,
                                         test_targets)
