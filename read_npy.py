import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Read .npy file.')

parser.add_argument('filename', type=str, help='The path to the .npy file')

args = parser.parse_args()

data = np.load(args.filename)
# np.set_printoptions(threshold=np.inf)
print(data.shape)
print(data.dtype)
print("data.max = ", data.max())
print(data[:,:,:50257])
#print(data[0][0][4096-256:4096])
#
