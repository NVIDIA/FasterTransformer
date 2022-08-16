# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from sacrebleu import corpus_bleu

def bleu_score(pred_file, ref_file, bleu_score_threshold=None):
    with open(pred_file, "r") as pred_stream, open(ref_file, "r") as ref_stream:
        pred_stream_txt = pred_stream.readlines()
        ref_stream_txt = ref_stream.readlines()
        bleu = corpus_bleu(pred_stream_txt, [ref_stream_txt], force=True)
        print("       bleu score: {:6.2f}".format(bleu.score))
        print("       bleu counts: {}".format(bleu.counts))
        print("       bleu totals: {}".format(bleu.totals))
        print("       bleu precisions: {}".format(bleu.precisions))
        print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
        if bleu_score_threshold != None:
            assert bleu.score >= bleu_score_threshold, "TEST FAIL !"
            print("[INFO] TEST PASS !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred_file', type=str, metavar='NUMBER',
                        help='The prediction files.', required=True)
    parser.add_argument('--ref_file', type=str, metavar='NUMBER',
                        help='The reference files.', required=True)
    parser.add_argument('--bleu_score_threshold', type=float, metavar='NUMBER',
                        help='The threshold of bleu score.')
    args = parser.parse_args()

    bleu_score(args.pred_file, args.ref_file, args.bleu_score_threshold)
