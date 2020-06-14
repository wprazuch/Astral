import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse


class TiffSplitter():

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path

    print(os.getcwd())

    def run(self, input_path=None, output_path=None):
        if input_path is not None:
            self.input_path = input_path
        if output_path is not None:
            self.output_path = output_path

        self.__check_output_path()

        img = Image.open(input_path)
        for i in range(1200):
            try:
                img.seek(i)
                img.save(os.path.join(output_path, 'image_%s.tif' % (i,)))
            except EOFError:
                return None

    def __check_output_path(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


def __main__():
    parser = argparse.ArgumentParser(prog='tiffsplitter')
    parser.add_argument('-f', '--filename', help='filename to add')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    input_file = args.filename
    rootdir = args.rootdir
    directory = input_file.split('.')[0]

    sequence_dir = os.path.join(rootdir, directory)
    if not os.path.exists(sequence_dir):
        os.makedirs(sequence_dir)
    input_path = os.path.join(rootdir, input_file)
    output_path = os.path.join(rootdir, directory, 'image_sequence')

    tiff_splitter = TiffSplitter()
    tiff_splitter.run(input_path, output_path)


if __name__ == '__main__':
    __main__()
