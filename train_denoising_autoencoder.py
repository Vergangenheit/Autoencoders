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

class TrainDenoisingAutoencoder:

    def __init__(self, EPOCHS=25, BS=32):
        self.EPOCHS = EPOCHS
        self.BS = BS
        self.trainX = None
        self.testX = None
        self.trainXNoisy = None
        self.testXNoisy = None
        self.H = None


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
        # sample noise from a random normal distribution centered at 0.5 (since
        # our images lie in the range [0, 1]) and a standard deviation of 0.5
        trainNoise = np.random.normal(loc=0.5, scale=0.5, size=self.trainX.shape)
        testNoise = np.random.normal(loc=0.5, scale=0.5, size=self.testX.shape)
        self.trainXNoisy = np.clip(self.trainX + trainNoise, 0, 1)
        self.testXNoisy = np.clip(self.testX + testNoise, 0, 1)

    def train(self):
        # construct our convolutional autoencoder
        print("[INFO] building autoencoder...")
        (encoder, decoder, self.autoencoder) = ConvAutoencoder.build(28, 28, 1)
        opt = Adam(lr=1e-3)
        self.autoencoder.compile(loss="mse", optimizer=opt)
        # train the convolutional autoencoder
        self.H = self.autoencoder.fit(
            self.trainXNoisy, self.trainX,
            validation_data=(self.testXNoisy, self.testX),
            epochs=self.EPOCHS,
            batch_size=self.BS)

    def plot(self, args):
        # construct a plot that plots and saves the training history
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

    def predict_print(self, args):
        # use the convolutional autoencoder to make predictions on the
        # testing images, then initialize our list of output images
        print("[INFO] making predictions...")
        decoded = self.autoencoder.predict(self.testXNoisy)
        outputs = None

        # loop over our number of output samples
        for i in range(0, args["samples"]):
            # grab the original image and reconstructed image
            original = (self.testXNoisy[i] * 255).astype("uint8")
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

        # save the outputs image to disk
        cv2.imwrite(args["output"], outputs)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--samples", type=int, default=8,
                    help="# number of samples to visualize when decoding")
    ap.add_argument("-o", "--output", type=str, default="output.png",
                    help="path to output visualization file")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output plot file")
    args = vars(ap.parse_args())
    train = TrainDenoisingAutoencoder()
    train.load_and_process()
    train.train()
    train.plot(args)
    train.predict_print(args)
