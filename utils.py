# -*- coding: utf-8 -*-
import os
import collections
import codecs
import cPickle
import re
import glob
import numpy as np

class TextLoader():
  def __init__(self, data_dir, batch_size, seq_length, lang="en"):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    main_input_file = os.path.join(data_dir, "input." + lang + ".txt")
    input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
      if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1] == ".txt"]
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_files = glob.glob(os.path.join(data_dir, "data.*.npy"))

    if not (os.path.exists(vocab_file)) and not len(tensor_files):
      print "reading text file"
      self.preprocess(main_input_file, input_files, vocab_file, tensor_files)
    else:
      print "loading preprocessed files"
      self.load_preprocessed(vocab_file, tensor_files, lang)
    self.create_batches()
    self.reset_batch_pointer()

  def preprocess(self, main_input_file, input_files, vocab_file, tensor_files):
    data = dict()
    langs = []

    with codecs.open(main_input_file, "r", "utf-8") as f:
      main_data = f.read()

    for input_file in input_files:
      lang = re.match(r".+\/input\.([a-z]+)\.txt", input_file).group(1)
      langs += [lang]

      with codecs.open(input_file, "r", "utf-8") as f:
        data[lang] = f.read()

    counter = collections.Counter("\n".join(data.values()))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.chars, _ = list(zip(*count_pairs))
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))

    with open(vocab_file, "w") as f:
      cPickle.dump(self.chars, f)

    self.tensor = np.array(map(self.vocab.get, main_data))

    for key in data.keys():
      tensor_filename = "{}.{}.{}".format(os.path.join(self.data_dir, "data"), key, "npy")
      np.save(tensor_filename, np.array(map(self.vocab.get, data[key])))

  def load_preprocessed(self, vocab_file, tensor_files, lang):
    with open(vocab_file) as f:
      self.chars = cPickle.load(f)
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))

    # get target dataset
    tensor_file = tensor_files[0]

    for tf in tensor_files:
      if "." + lang + ".npy" in tf:
        tensor_file = tf

    self.tensor = np.load(tensor_file)
    self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

  def create_batches(self):
    self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
    self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.tensor
    ydata = np.copy(self.tensor)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
    self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

  def next_batch(self):
    x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
    self.pointer += 1
    return x, y

  def reset_batch_pointer(self):
    self.pointer = 0
