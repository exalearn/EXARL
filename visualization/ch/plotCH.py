import os
import argparse
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
matplotlib.rcParams.update({'font.size': 15})


fig      = plt.figure(1, figsize=(20, 10))
axImage  = fig.add_subplot(121)
axStruct = fig.add_subplot(122)

M = 128  # size of image
N = 128


def parse_args():

    parser = argparse.ArgumentParser(description='arguments for the CH plots')

    parser.add_argument('-dir_file', default='./sample_data', type=str,
                        help='Specify the output directory', required=False)

    parser.add_argument('-step', default=99, type=int, help='time step',
                        required=False)

    return parser.parse_args()


# plot 2D image
def plotCH2D(args):

    data_file = os.path.join(args.dir_file, 'C_' + str(args.step) + '.out')
    C = np.genfromtxt(data_file)
    C = np.reshape(C, [M, N])

    axImage.cla()
    axImage.contourf(C, 30, vmin=-1, vmax=1)
    axImage.set_aspect('equal')
    axImage.set_title("2D CH image")  # set a title


# plot structure vector
def plotStructVector(args):

    data_file = os.path.join(args.dir_file,
                             'circfft_' + str(args.step) + '_fft_circavg.out')
    sv_values = np.genfromtxt(data_file)[1]

    axStruct.cla()
    axStruct.plot(sv_values, color='blue', marker='.')
    axStruct.set_title('Structure Vector')
    axStruct.set_xlabel('Conponent of Structure Vector')
    axStruct.set_aspect('auto', 'box')


def main():

    args = parse_args()

    plotCH2D(args)
    plotStructVector(args)

    figName = 'CH_' + str(args.step) + '.png'
    fig.savefig(figName, bbox_inches='tight')


if __name__ == "__main__":
    main()
