# guild run shape_keras.py epoch=[50] learning-rate=[0.1,0.01,0.001] log-dir=runs/shapes_002
# guild tensorboard --host 0.0.0.0 --port 6006

import os
import glob
import matplotlib
import datetime
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback

import matplotlib.cm as cm

import tensorflow as tf

from deeplearn.keras_models import simple_model, compile_model
from deeplearn.keras_helpers import make_gradcam_heatmap, log_confusion_matrix

####################################################
class ShapeClassifier:
    ####################################################
    def __init__(self, model_file=None, data_path=None, logdir="runs/pets"):
        self.model_file = model_file
        self.model = None        
        self.dataPath = data_path
        self.class_names = ["cat", "dog"]
        self.training_set = None
        self.valid_set = None
        self.NUM_SAVE_IMAGES = 5

        self.file_writer = tf.summary.create_file_writer(logdir)
        self.tensorboard_callback = TensorBoard(log_dir=logdir,
                                                histogram_freq=1,
                                                write_images=True)
        self.cm_callback = LambdaCallback(on_epoch_end=self.gen_confusion_matrix)

    ####################################################
    def load_data(self, BATCH_SIZE=8):
        # Load dataset
        train_datagen = ImageDataGenerator(rescale = 1./255)
        valid_datagen = ImageDataGenerator(rescale = 1./255)
        self.training_set = train_datagen.flow_from_directory(os.path.join(self.dataPath, 'train'),
                                                        target_size = (224, 224),
                                                        batch_size = BATCH_SIZE,
                                                        class_mode = 'categorical')
        self.valid_set = valid_datagen.flow_from_directory(os.path.join(self.dataPath, 'train'),
                                                    target_size = (224, 224),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical')

    ####################################################
    def gen_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the valid_images.
        valid_fname = []
        valid_images = []
        valid_labels = []
        for img_class in self.class_names:
            for img_fname in glob.glob(os.path.join(self.dataPath, "valid/%s/*.png"%(img_class))):
                img = image.load_img(img_fname, target_size=(224, 224))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = np.array(img, dtype="float") / 255.0
                valid_fname.append(img_fname)
                valid_images.append(img)
                valid_labels.append(self.class_names.index(img_class))

        valid_images = np.vstack(valid_images)
        valid_pred = self.model.predict_classes(valid_images)

        log_confusion_matrix(valid_labels, valid_pred, self.class_names, epoch, logs, self.file_writer)

    ####################################################
    def train(self, BATCH_SIZE=8, NUM_EPOCHS=10):
        checkpointer = ModelCheckpoint(filepath=self.model_file, 
                                    monitor='accuracy',
                                    verbose=1, 
                                    save_best_only=True,
                                    mode='auto')


        TRAIN_STEPS_PER_EPOCH = np.ceil((self.training_set.n/BATCH_SIZE)-1)
        VAL_STEPS_PER_EPOCH = np.ceil((self.valid_set.n/BATCH_SIZE)-1)

        history = self.model.fit_generator(self.training_set,
                                        steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
                                        epochs = NUM_EPOCHS,
                                        callbacks=[checkpointer, self.tensorboard_callback, self.cm_callback],
                                        validation_data = self.valid_set,
                                        validation_steps = VAL_STEPS_PER_EPOCH,
                                        )

    ####################################################
    def eval_image(self, model):
        self.model.load_weights(self.model_file)
        # Use the model to predict the values from the valid_images.
        valid_fname = []
        valid_images = []
        valid_labels = []
        for img_class in self.class_names:
            for img_fname in glob.glob(os.path.join(self.dataPath, "valid/%s/*.png"%(img_class))):
                img = image.load_img(img_fname, target_size=(224, 224))
                img_tensor = image.img_to_array(img)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor /= 255.
                valid_fname.append(img_fname)
                valid_images.append(img_tensor)
                valid_labels.append(self.class_names.index(img_class))

        valid_image_stack = np.vstack(valid_images)
        valid_pred = self.model.predict_classes(valid_image_stack)

        saved_images = 0
        for idx in range(len(valid_fname)):
            fname = valid_fname[idx]
            prediction = valid_pred[idx]
            label = valid_labels[idx]
            # Compare prediciton to label
            if prediction != label:
                print("File: %s Label: %s Pred: %s" % (fname, prediction, label))
                # Save failed images
                if saved_images<=self.NUM_SAVE_IMAGES:
                    img_tensor = valid_images[idx]
                    saved_images += 1
                    # Using the file writer, log the reshaped image.
                    with self.file_writer.as_default():
                        tf.summary.image("Predict: %s : %s"%(fname, prediction),  img_tensor, step=0)

                    # Generate class activation heatmap
                    last_conv_layer_name = "conv2d_5"
                    classifier_layer_names = ["flatten"]
                    heatmap = make_gradcam_heatmap(
                        img_tensor, self.model, last_conv_layer_name, classifier_layer_names
                    )

                    # We rescale heatmap to a range 0-255
                    heatmap = np.uint8(255 * heatmap)

                    # We use jet colormap to colorize heatmap
                    jet = cm.get_cmap("jet")

                    # We use RGB values of the colormap
                    jet_colors = jet(np.arange(256))[:, :3]
                    jet_heatmap = jet_colors[heatmap]

                    # We create an image with RGB colorized heatmap
                    jet_heatmap = image.array_to_img(jet_heatmap)
                    jet_heatmap = jet_heatmap.resize((img_tensor.shape[1], img_tensor.shape[0]))
                    jet_heatmap = image.img_to_array(jet_heatmap)

                    # Superimpose the heatmap on original image
                    superimposed_img = jet_heatmap * 0.4 + img_tensor

                    # Using the file writer, log the reshaped image.
                    with self.file_writer.as_default():
                        tf.summary.image("heatmap: %s : %s"%(fname, prediction), superimposed_img, step=0)

####################################################
def main(args):
    s = ShapeClassifier(model_file=args.model_file, 
                        data_path=args.data_path, 
                        logdir=args.log_dir)

    s.model = simple_model(input_size=(224, 224, 3), output_size=2)
    compile_model(
        s.model,
        optimizer=args.optimizer, 
        learning_rate=args.learning_rate,
    )
    s.load_data(args.batch_size)
    s.train(BATCH_SIZE=args.batch_size, 
            NUM_EPOCHS=args.epoch)
    # valid Image
    s.eval_image(s.model)

    
####################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-file", type=str, default="best_weights.hdf5")
    parser.add_argument("-d", "--data-path", type=str, default="/home/jrm/workspace/deeplearn/datasets/pets/")
    parser.add_argument("-o", "--optimizer", type=str, default="RMSprop")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-r", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-l", "--log-dir", type=str, default=os.getenv("TENSORBOARD_LOGS", "runs/pets"))
    args = parser.parse_args()

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    print(sess)

    main(args)

    