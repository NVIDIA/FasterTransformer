# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

# usage example
# python ./convertInput.py -t $data_name -b $batch_size -d $data_dir  -l $seq_len -s $spiece_model_file -o $data_file -u 0

from os.path import join
from absl import flags
import os
import sys
import getopt
import csv
import collections
import numpy as np
import json
import random
from copy import copy
from collections import defaultdict as dd
import numpy as np

import six
import unicodedata
import sentencepiece as spm

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

SEP_ID = special_symbols["<sep>"]
CLS_ID = special_symbols["<cls>"]
SPIECE_UNDERLINE = '▁'


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


"""
DataProcessor Classes
"""


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) == 0:
                    continue
                lines.append(line)
            return lines


class GLUEProcessor(DataProcessor):
    def __init__(self):
        self.train_file = "train.tsv"
        self.dev_file = "dev.tsv"
        self.test_file = "test.tsv"
        self.label_column = None
        self.text_a_column = None
        self.text_b_column = None
        self.contains_header = True
        self.test_text_a_column = None
        self.test_text_b_column = None
        self.test_contains_header = True

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.train_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.dev_file)), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        if self.test_text_a_column is None:
            self.test_text_a_column = self.text_a_column
        if self.test_text_b_column is None:
            self.test_text_b_column = self.text_b_column

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = "%s-%s" % (set_type, i)

            a_column = (self.text_a_column if set_type != "test" else
                        self.test_text_a_column)
            b_column = (self.text_b_column if set_type != "test" else
                        self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                tf.logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = self.get_labels()[0]
            else:
                if len(line) <= self.label_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                label = line[self.label_column]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Yelp5Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"))

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, input_file):
        """Creates examples for the training and dev sets."""
        examples = []
        with tf.gfile.Open(input_file) as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):

                label = line[0]
                text_a = line[1].replace('""', '"').replace('\\"', '"')
                examples.append(
                    InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    def get_labels(self):
        return ["neg", "pos"]

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test"))

    def _create_examples(self, data_dir):
        examples = []
        for label in ["neg", "pos"]:
            cur_dir = os.path.join(data_dir, label)
            for filename in tf.gfile.ListDirectory(cur_dir):
                if not filename.endswith("txt"):
                    continue

                path = os.path.join(cur_dir, filename)
                with tf.gfile.Open(path) as f:
                    text = f.read().strip().replace("<br />", " ")
                examples.append(InputExample(
                    guid="unused_id", text_a=text, text_b=None, label=label))
        return examples


class MnliMatchedProcessor(GLUEProcessor):
    def __init__(self):
        super(MnliMatchedProcessor, self).__init__()
        self.dev_file = "dev_matched.tsv"
        self.test_file = "test_matched.tsv"
        self.label_column = -1
        self.text_a_column = 8
        self.text_b_column = 9

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]


class MnliMismatchedProcessor(MnliMatchedProcessor):
    def __init__(self):
        super(MnliMismatchedProcessor, self).__init__()
        self.dev_file = "dev_mismatched.tsv"
        self.test_file = "test_mismatched.tsv"


class StsbProcessor(GLUEProcessor):
    def __init__(self):
        super(StsbProcessor, self).__init__()
        self.label_column = 9
        self.text_a_column = 7
        self.text_b_column = 8

    def get_labels(self):
        return [0.0]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and self.contains_header and set_type != "test":
                continue
            if i == 0 and self.test_contains_header and set_type == "test":
                continue
            guid = "%s-%s" % (set_type, i)

            a_column = (self.text_a_column if set_type != "test" else
                        self.test_text_a_column)
            b_column = (self.text_b_column if set_type != "test" else
                        self.test_text_b_column)

            # there are some incomplete lines in QNLI
            if len(line) <= a_column:
                tf.logging.warning('Incomplete line, ignored.')
                continue
            text_a = line[a_column]

            if b_column is not None:
                if len(line) <= b_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                text_b = line[b_column]
            else:
                text_b = None

            if set_type == "test":
                label = self.get_labels()[0]
            else:
                if len(line) <= self.label_column:
                    tf.logging.warning('Incomplete line, ignored.')
                    continue
                label = float(line[self.label_column])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

# Tokenize


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    # return_unicode is used only for py2
    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, ''))
            if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


# Convert functions
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenize_fn):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[1] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    if label_list is not None:
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

    tokens_a = tokenize_fn(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenize_fn(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for two [SEP] & one [CLS] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]

    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(SEG_ID_B)
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_B)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens

    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if label_list is not None:
        label_id = label_map[example.label]
    else:
        label_id = example.label

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def file_based_convert_examples_to_npz(
        examples, label_list, max_seq_length, tokenize_fn, output_file,
        num_passes=1):
    """Convert a set of `InputExample`s to a NPZ file."""

    if num_passes > 1:
        examples *= num_passes

    data = {}
    arr_input_ids = []
    arr_input_mask = []
    arr_segment_ids = []
    arr_label_ids = []

    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenize_fn)

        arr_input_ids.append(feature.input_ids)
        arr_input_mask.append(feature.input_mask)
        arr_segment_ids.append(feature.segment_ids)
        arr_label_ids.append(feature.label_id)

        # if ex_index % 100 == 0:
        #  print("Writing example {} of {} with {} {} {} {}".format(ex_index,len(examples),feature.input_ids,
        #      feature.input_mask, feature.segment_ids, feature.label_id))

    data["input_ids:0"] = np.array(arr_input_ids, dtype=np.int32)
    data["input_mask:0"] = np.array(arr_input_mask, dtype=np.float32)
    data["segment_ids:0"] = np.array(arr_segment_ids, dtype=np.int32)
    data["label_ids:0"] = np.array(arr_label_ids, dtype=np.int32)

    print("Save Input to file {}".format(output_file))
    np.savez(output_file, **data)


def usage():
    print(" -t task_name")
    print(" -b batch_size")
    print(" -d data_dir")
    print(" -r is_regression")
    print(" -l max_seq_length")
    print(" -s spiece_model_file")
    print(" -o output_file")
    print(" -u uncased")
    print("Example: python convertInput.py -t sts-b -b 8 -d ../../../Data/glue_data/STS-B -r 1 -l 128 -s ../../../Data/xlnet_cased_L-12_H-768_A-12/spiece.model -o ./data.npz -u 0 ")


if __name__ == "__main__":
    # Input parameters
    task_name = "sts-b"
    batch_size = 8
    data_dir = "../../../Data/glue_data/STS-B/"
    is_regression = True
    max_seq_length = 128
    spiece_model_file = "../../../Data/xlnet_cased_L-12_H-768_A-12/spiece.model"
    output_file = "./data.npz"
    uncased = False

    # Set perameter
    opts, args = getopt.getopt(sys.argv[1:], "ht:b:d:r:l:s:o:u:")
    for op, value in opts:
        if op == "-t":
            task_name = value
        elif op == "-b":
            batch_size = int(value)
        elif op == "-d":
            data_dir = value
        elif op == "-r":
            is_regression = bool(value)
        elif op == "-l":
            max_seq_length = int(value)
        elif op == "-s":
            spiece_model_file = value
        elif op == "-o":
            output_file = value
        elif op == "-u":
            uncased = bool(value)
        elif op == "-h":
            usage()
            sys.exit()

    # Set processor
    processors = {
        "mnli_matched": MnliMatchedProcessor,
        "mnli_mismatched": MnliMismatchedProcessor,
        'sts-b': StsbProcessor,
        'imdb': ImdbProcessor,
        "yelp5": Yelp5Processor
    }
    processor = processors[task_name]()
    label_list = processor.get_labels() if not is_regression else None

    # Acquire examples
    eval_examples = processor.get_test_examples(data_dir)
    while len(eval_examples) % batch_size != 0:
        eval_examples.append(PaddingInputExample())

    # Convert examples to numpy
    sp = spm.SentencePieceProcessor()
    sp.Load(spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=uncased)
        return encode_ids(sp, text)

    file_based_convert_examples_to_npz(eval_examples, label_list,
                                       max_seq_length, tokenize_fn, output_file)

    #np.save('extra.npy', extra.transpose((1, 0, 2)))
