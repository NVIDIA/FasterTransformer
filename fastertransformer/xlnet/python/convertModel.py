from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging as _logging    # pylint: disable=unused-import


import tensorflow as tf
import numpy as np
import getopt 
import sys


def usage():
    print(" -o output_file")
    print(" -i ckpt_dir")
    print("Example: python convertModel.py -i xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt -o ./model.npz ")


if __name__=="__main__":
    m = {}
    ckpt_dir="../../../Data/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt"
    output_file="./model.npz"

    #Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:") 
    for op, value in opts:
        if op == "-i":
            ckpt_dir = value
        if op == "-o":
            output_file = value
        if op == "-h":
            usage()
            sys.exit()

    saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_dir))
    with tf.Session() as sess:
        saver.restore(sess,ckpt_dir)

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        idx=0
        for var in all_vars:
            m[var.name] = sess.run(var)
            print (str(idx) + " " + str(var.name) + " " + str(var.shape))
            #print(m[var.name].flatten()[:10])
            idx += 1

    np.savez(output_file, **m)
 

