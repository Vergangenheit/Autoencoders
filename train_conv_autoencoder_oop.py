# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


class TrainConvAutoencoder:

    def __init__(self, EPOCHS=25, BS=30):
        self.EPOCHS = EPOCHS
        self.BS = BS
        self.trainX = None
        self.testX = None
        self.H = None
        self.autoencoder = None

    def load_and_process(self):
        # load the MNIST dataset
        print("[INFO] loading MNIST dataset...")
        ((self.trainX, _), (self.testX, _)) = mnist.load_data()

        # add a channel dimension to every image in the dataset, then scale
        # the pixel intensities to the range [0, 1]
        self.trainX = np.expand_dims(self.trainX, axis=-1)
        self.testX = np.expand_dims(self.testX, axis=-1)
        self.trainX = self.trainX.astype("float32") / 255.0
        self.testX = self.testX.astype("float32") / 255.0

    def train(self):
        # construct our convolutional autoencoder
        print("[INFO] building autoencoder...")
        (encoder, decoder, self.autoencoder) = ConvAutoencoder.build(28, 28, 1)
        opt = Adam(lr=1e-3)
        self.autoencoder.compile(loss="mse", optimizer=opt)

        # train the convolutional autoencoder
        self.H = self.autoencoder.fit(self.trainX, self.trainX,
                                      validation_data=(self.testX, self.testX),
                                      epochs=self.EPOCHS,
                                      batch_size=self.BS)

    def plot(self, args):
        N = np.arange(0, self.EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, self.H.history["loss"], label="train_loss")
        plt.plot(N, self.H.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])

    def predict_and_print(self, args):
        # use the convolutional autoencoder to make predictions on the
        # testing images, then initialize our list of output images
        print("[INFO] making predictions...")
        decoded = self.autoencoder.predict(self.testX)
        outputs = None

        # loop over our number of output samples
        for i in range(0, args["samples"]):
            # grab the original image and reconstructed image
            original = (self.testX[i] * 255).astype("uint8")
            recon = (decoded[i] * 255).astype("uint8")

            # stack the original and reconstructed image side-by-side
            output = np.hstack([original, recon])

            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            if outputs is None:
                outputs = output

            # otherwise, vertically stack the outputs
            else:
                outputs = np.vstack([outputs, output])

        cv2.imwrite(args["outputs"], outputs)
