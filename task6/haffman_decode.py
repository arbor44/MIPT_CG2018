import pickle
import argparse
from haffman_utils import decode

parser = argparse.ArgumentParser()
parser.add_argument('input', help='path to input binary file.')
parser.add_argument('output', help='path to the output file.')
args = parser.parse_args()

with open(args.input, 'rb') as in_, open(args.output, 'wb') as out:
    out.write(decode(*pickle.load(in_)))