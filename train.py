from train_conv_autoencoder_oop import TrainConvAutoencoder
import argparse

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--samples", type=int, default=8,
                    help="# number of samples to visualize when decoding")
    ap.add_argument("-o", "--output", type=str, default="output.png",
                    help="path to output visualization file")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output plot file")
    args = vars(ap.parse_args())
    tr_autoenc = TrainConvAutoencoder()
    tr_autoenc.load_and_process()
    tr_autoenc.train()
    tr_autoenc.plot(args)
    tr_autoenc.predict_and_print(args)


if __name__ == '__main__':
    main()
