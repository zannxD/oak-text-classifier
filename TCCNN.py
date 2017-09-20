import urllib2
import numpy as np
import re
import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Pull sentences with positive sentiment
    pos_file = urllib2.urlopen('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos')

    # Pull sentences with negative sentiment
    neg_file = urllib2.urlopen('https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg')

    # Load data from files
    positive_examples = list(pos_file.readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(neg_file.readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

"""
Loads and preprocessed data for the MR dataset.
Returns input vectors, labels, vocabulary, and inverse vocabulary.
"""
# Load and preprocess data
sentences, labels = load_data_and_labels()
sentences_padded = pad_sentences(sentences)
vocabulary, vocabulary_inv = build_vocab(sentences_padded)
x, y = build_input_data(sentences_padded, labels, vocabulary)

vocab_size = len(vocabulary)

# randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# split train/dev set
# there are a total of 10662 labeled examples to train on
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

sentence_size = x_train.shape[1]

print ('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
print ('train shape:', x_train.shape)
print ('dev shape:', x_dev.shape)
print ('vocab_size', vocab_size)
print ('sentence max words', sentence_size)
print("======================Data Preprocessed======================================")

import mxnet as mx
import sys,os

'''
Define batch size and the place holders for network inputs and outputs
'''

batch_size = 50 # the size of batches to train network with
print ('batch size', batch_size)

input_x = mx.sym.Variable('data') # placeholder for input data
input_y = mx.sym.Variable('softmax_label') # placeholder for output label


'''
Define the first network layer (embedding)
'''

# create embedding layer to learn representation of words in a lower dimensional subspace (much like word2vec)
num_embed = 300 # dimensions to embed words into
print ('embedding dimensions', num_embed)

embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

# reshape embedded data for next layer
conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))
print ("===========================================================")

# create convolution + (max) pooling layer for each filter operation
filter_list=[3, 4, 5] # the size of filters to use
print ('convolution filters', filter_list)

num_filter=100
pooled_outputs = []
for i, filter_size in enumerate(filter_list):
    convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
    relui = mx.sym.Activation(data=convi, act_type='relu')
    pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
    pooled_outputs.append(pooli)

# combine all pooled outputs
total_filters = num_filter * len(filter_list)
concat = mx.sym.Concat(*pooled_outputs, dim=1)

# reshape for next layer
h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

print("==============================================================")

# dropout layer
dropout=0.5
print ('dropout probability', dropout)

if dropout > 0.0:
    h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
else:
    h_drop = h_pool
    
    
print("===============================================================")

# fully connected layer
num_label=2

cls_weight = mx.sym.Variable('cls_weight')
cls_bias = mx.sym.Variable('cls_bias')

fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

# softmax output
sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')
print('sm: ', sm)

# set CNN pointer to the "back" of the network
cnn = sm


print("===============================================================")

from collections import namedtuple
import time
import math

# Define the structure of our CNN Model (as a named tuple)
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

# Define what device to train/test on
ctx=mx.cpu(0)
# If you have no GPU on your machine change this to
# ctx=mx.cpu(0)

arg_names = cnn.list_arguments()

input_shapes = {}
input_shapes['data'] = (batch_size, sentence_size)

arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
args_grad = {}
for shape, name in zip(arg_shape, arg_names):
    if name in ['softmax_label', 'data']: # input, output
        continue
    args_grad[name] = mx.nd.zeros(shape, ctx)

cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

param_blocks = []
arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
initializer=mx.initializer.Uniform(0.1)
for i, name in enumerate(arg_names):
    if name in ['softmax_label', 'data']: # input, output
        continue
    initializer(name, arg_dict[name])

    param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

data = cnn_exec.arg_dict['data']
label = cnn_exec.arg_dict['softmax_label']

cnn_model= CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)



print("===============================================================")

'''
Train the cnn_model using back prop
'''

optimizer='rmsprop'
max_grad_norm=5.0
learning_rate=0.0005
epoch=50

print ('optimizer', optimizer)
print ('maximum gradient', max_grad_norm)
print ('learning rate (step size)', learning_rate)
print ('epochs to train for', epoch)

# create optimizer
opt = mx.optimizer.create(optimizer)
opt.lr = learning_rate

updater = mx.optimizer.get_updater(opt)

# create logging output
logs = sys.stderr
print("For each training epoch")
# For each training epoch
for iteration in range(epoch):
    tic = time.time()
    num_correct = 0
    num_total = 0

    # Over each batch of training data
    for begin in range(0, x_train.shape[0], batch_size):
        batchX = x_train[begin:begin+batch_size]
        batchY = y_train[begin:begin+batch_size]
        if batchX.shape[0] != batch_size:
            continue

        cnn_model.data[:] = batchX
        cnn_model.label[:] = batchY

        # forward
        cnn_model.cnn_exec.forward(is_train=True)

        # backward
        cnn_model.cnn_exec.backward()

        # eval on training data
        num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

        # update weights
        norm = 0
        for idx, weight, grad, name in cnn_model.param_blocks:
            grad /= batch_size
            l2_norm = mx.nd.norm(grad).asscalar()
            norm += l2_norm * l2_norm

        norm = math.sqrt(norm)
        for idx, weight, grad, name in cnn_model.param_blocks:
            if norm > max_grad_norm:
                grad *= (max_grad_norm / norm)

            updater(idx, grad, weight)

            # reset gradient to zero
            grad[:] = 0.0

    # Decay learning rate for this epoch to ensure we are not "overshooting" optima
    if iteration % 50 == 0 and iteration > 0:
        opt.lr *= 0.5
        print >> logs, 'reset learning rate to %g' % opt.lr

    # End of training loop for this epoch
    toc = time.time()
    train_time = toc - tic
    train_acc = num_correct * 100 / float(num_total)

    # Saving checkpoint to disk
    if (iteration + 1) % 10 == 0:
        prefix = 'cnn'
        cnn_model.symbol.save('./%s-symbol.json' % prefix)
        save_dict = {('arg:%s' % k) :v  for k, v in cnn_model.cnn_exec.arg_dict.items()}
        save_dict.update({('aux:%s' % k) : v for k, v in cnn_model.cnn_exec.aux_dict.items()})
        param_name = './%s-%04d.params' % (prefix, iteration)
        mx.nd.save(param_name, save_dict)
        print >> logs, 'Saved checkpoint to %s' % param_name


    # Evaluate model after this epoch on dev (test) set
    num_correct = 0
    num_total = 0

    # For each test batch
    for begin in range(0, x_dev.shape[0], batch_size):
        batchX = x_dev[begin:begin+batch_size]
        batchY = y_dev[begin:begin+batch_size]

        if batchX.shape[0] != batch_size:
            continue

        cnn_model.data[:] = batchX
        cnn_model.cnn_exec.forward(is_train=False)

        num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
        num_total += len(batchY)

    dev_acc = num_correct * 100 / float(num_total)
    print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
            --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc)

#cnn_model.predict()


