from __future__ import division, print_function

import json
import logging
import h5py
import codecs
import os
import sys
import getopt
import numpy as np

import h5tojson
from h5json import Hdf5db

def parseCommandLine(argv):
    def printUsage():
        print("""
    Usage:
      python Keras2NCNN.py -i inputFile

    Example:
      python .\\Keras2NCNN.py -i .\\keras_model.h5

    Parameters:
        -i: input .h5 format Keras model
        """)

    model_input = ""
    try:
        (opts,args) = getopt.getopt(argv,"i:")
        if len(opts) == 0:
            printUsage()
            sys.exit(1)
        for (opt, arg) in opts:
            if opt == "-i":
                model_input = arg
    except:
        printUsage()
        sys.exit(1)
    return model_input

def initial_logger():
    logger = logging.getLogger("Keras2NCNN")
    logger.setLevel(logging.DEBUG)
    log_path = "{}{}.log".format(".\\", "Keras2NCNN")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(filename)s LINE-%(lineno)d | PROCESS-%(process)d | %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

class LayerParameter_ncnn (object):
    def __init__(self):
        self.name = ''
        self.type = ''
        self.param = []
        self.weights = []

def find_weights_root(h5, layerName):
    """
       recursively find the weights and biases value of h5 format layer.

       For example,

       dense weights root:
        "/time_distributed_1/custom_blstm/time_distributed_1".

       blstm weights root:
        "/bidirectional_1/custom_blstm/bidirectional_1/forward_lstm_1/bias:0"
        "/bidirectional_1/custom_blstm/bidirectional_1/backward_lstm_1/bias:0"
    """
    layer = h5
    layer_name = layerName
    while True:
        layer = layer[layer_name]
        if (not hasattr(layer, "keys")) or len(layer.keys()) > 1:
            break
        layer_name = list(layer.keys())[0]
    return layer

def relu_dump(h5, layerName):
    layer = LayerParameter_ncnn()
    layer.name = 'relu_' + layerName
    layer.type = 'ReLU'

    return layer

def softmax_dump(h5, layerName):
    layer = LayerParameter_ncnn()
    layer.name = 'prob_' + layerName
    layer.type = 'Softmax'
    # axis
    layer.param.append('%d' % 0)

    return layer

def dense_dump(h5, layerName):
    dense = find_weights_root(h5, layerName)
    assert len(dense.keys()) == 2, "expected to have two elements: dense weights, dense biases; but got: {}".format(
        dense.keys())

    kernel = dense['kernel:0']
    bias = dense['bias:0']
    npkernel = np.asarray(kernel)
    npkernel = np.transpose(npkernel)
    npbias = np.asarray(bias)


    thefile = open('bias.txt', 'w')

    for item in npbias:
        thefile.write("%s\n" % item)
    thefile.close()

    sys.exit(1)

    layer = LayerParameter_ncnn()
    layer.name = 'fc_' + layerName
    layer.type = 'InnerProduct'
    layer.param.append('%d' % 300)
    layer.param.append('%d' % True)
    layer.param.append('%d' % npkernel.size)

    layer.weights.append(np.array([0.]))
    layer.weights.append(npkernel)
    layer.weights.append(np.array([0.]))
    layer.weights.append(npbias)

    return layer

def bn_dump(h5, layerName, epsilon=0.001):
    '''
    slope mean variance bias
    gamma mean variance beta

    Extract batch normalization (BN) weights mean, var, beta and gamma,
    and transform the BN weights to kernel and bias.

    output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta
           = (x * kernel) + bias

    kernel = (1 / (sqrt(var) + epsilon)) * gamma
    bias = beta - (1 / (sqrt(var) + epsilon)) * gamma * mean

    Note: for model inference, mean and var should be pre-computed on training set (NOT on test set)
    Reference: http://arxiv.org/abs/1502.03167
    '''
    batchnorm = find_weights_root(h5, layerName)
    assert len(batchnorm.keys()) == 4, "expected to have four elements in batchnorm; but got: {}".format(
        batchnorm.keys())

    moving_mean = batchnorm['moving_mean:0']
    moving_variance = batchnorm['moving_variance:0']
    gamma = batchnorm['gamma:0']
    beta = batchnorm['beta:0']

    npmoving_mean = np.asarray(moving_mean)
    npmoving_variance = np.asarray(moving_variance)
    npgamma = np.asarray(gamma)
    npbeta = np.asarray(beta)
    npepsilon = np.full(npmoving_variance.shape, epsilon)

    npkernel = np.multiply(np.reciprocal(np.sqrt(np.add(npmoving_variance, npepsilon))), npgamma)
    npbias = np.subtract(npbeta, np.multiply(npmoving_mean, npkernel))

    layer = LayerParameter_ncnn()
    layer.name = 'bn_' + layerName
    layer.type = 'Scale'
    layer.param.append('%d' % npkernel.size)
    layer.param.append('%d' % 1)

    layer.weights.append(np.array([0.]))
    layer.weights.append(npkernel)
    layer.weights.append(np.array([0.]))
    layer.weights.append(npbias)

    return layer

def embedding_dump(h5, layerName, outfn):
    embeddings = find_weights_root(h5, layerName)

    embeddings_dic = {}
    for idx, embedding in enumerate(embeddings):
        embedding_str = " ".join(str(x) for x in embedding)
        embeddings_dic[idx] = embedding_str

    outfn = outfn + ".json"
    with codecs.open(outfn, 'w', "utf8") as fh:
        json.dump(embeddings_dic, fh, ensure_ascii=False)

def lstm_dump(h5, layerName, logger=None):
    layer = find_weights_root(h5, layerName)
    print(list(layer))
    assert len(layer.keys()) == 3

    rec = layer['recurrent_kernel:0']
    kernel = layer['kernel:0']
    bias = layer['bias:0']

    npkernel = np.asarray(kernel)
    print(npkernel.size)
    nprec = np.asarray(rec)
    print(nprec.size)
    npbias = np.asarray(bias)
    print(npbias.size)

    layer = LayerParameter_ncnn()
    layer.name = 'lstm_' + layerName
    layer.type = 'LSTM'
    layer.param.append('%d' % 300)
    layer.param.append('%d' % npkernel.size)

    layer.weights.append(np.array([0.]))
    layer.weights.append(nprec)
    layer.weights.append(npkernel)
    layer.weights.append(npbias)

    return layer

def pooling_dump(h5, layerName):
    layer = LayerParameter_ncnn()
    layer.name = 'pool_' + layerName
    layer.type = 'Pooling'

    # pooling_type: PoolMethod_MAX = 0, PoolMethod_AVE = 1
    layer.param.append('%d' % 1)
    # kernel_w: default = 0
    layer.param.append('%d' % 0)
    # stride_w: default = 1
    layer.param.append('%d' % 1)
    # pad_left: default = 0
    layer.param.append('%d' % 0)
    # global_pooling: True = 1, False = 0
    layer.param.append('%d' % 1)
    # pad_mode: default = 0
    layer.param.append('%d' % 0)

    return layer

class Converter(object):
    def __init__(self, h5_weights, logger=None):
        assert os.path.isfile(h5_weights), "file {} not exist!".format(h5_weights)
        self.h5_file = h5_weights
        self.h5 = h5py.File(h5_weights, 'r')
        self.logger = logger
        self.isNewFormat = None

    def _get_layers(self):

        dbFilename = h5tojson.getTempFileName()
        h5_json = ""
        with Hdf5db(self.h5_file, dbFilePath=dbFilename, readonly=True) as db:
            dumper = h5tojson.DumpJson(db)
            h5_json = dumper.dumpFile()

        # find root node according to name
        # for latest keras like 2.1.2, root node's alias = /model_weights
        # for version before 2.1.2, root node's alias = /
        # check it by order

        self.isNewFormat = False
        for _, v in h5_json["groups"].items():
            if v["alias"][0] == "/model_weights":
                self.isNewFormat = True
                break
        if self.isNewFormat is None:
            for _, v in h5_json["groups"].items():
                if v["alias"][0] == "/":
                    self.isNewFormat = False
                    break

        if self.isNewFormat is None:
            self.logger.error("Cannot found root node in h5 file, return")
            return

        layers = None

        for _, v in h5_json["groups"].items():
            if self.isNewFormat is True:
                if v["alias"][0] == "/model_weights":
                    layers = v["attributes"][0]["value"]
            else:
                if v["alias"][0] == "/":
                    layers = v["attributes"][0]["value"]

        return layers

    def _dump_layer_weights(self, layer_name):

        if self.isNewFormat is True:
            h5 = self.h5['model_weights']
        else:
            h5 = self.h5

        layer_rtn = []

        # layers
        if "dense" in layer_name:
            layer_rtn.append(dense_dump(h5, layer_name))
            layer_rtn.append(relu_dump(h5, layer_name))
            return layer_rtn
        elif "norm" in layer_name:
            layer_rtn.append(bn_dump(h5, layer_name))
            return layer_rtn
        elif "LSTM" in layer_name:
            layer_rtn.append(lstm_dump(h5, layer_name, self.logger))
            return layer_rtn
        elif "pooling" in layer_name:
            layer_rtn.append(pooling_dump(h5, layer_name))
            return layer_rtn

        # output
        elif "topic_class" in layer_name:
            layer_rtn.append(dense_dump(h5, layer_name))
            layer_rtn.append(softmax_dump(h5, layer_name))
            return layer_rtn

        # input
        elif "embedding" in layer_name:
            outfn = "{}.{}.{}".format(self.h5_file, layer_name, "fi_weights")
            embedding_dump(h5, layer_name, outfn)
            return None

        elif "flatten" in layer_name:
            self.logger.debug("found flatten layer.")
            return None
        elif "repeat" in layer_name:
            self.logger.debug("found repeat layer.")
            return None
        elif "input" in layer_name:
            self.logger.debug("found input layer.")
            return None
        elif "masking" in layer_name:
            self.logger.debug("found masking layer.")
            return None
        else:
            raise ValueError("non supported layer:{}".format(layer_name))

    def link_ncnn(self, layer, ncnn_net, ncnn_weights):
        layer_type = layer.type
        layer_param = layer.param
        if isinstance(layer_param, list):
            for ind, param in enumerate(layer_param):
                layer_param[ind] = str(ind) + '=' + param
        elif isinstance(layer_param, dict):
            param_dict = layer_param
            layer_param = []
            for key, param in param_dict.items():
                layer_param.append(key + '=' + param)

        pp = []
        pp.append('%-16s' % layer_type)
        pp.append('%-16s %d %d' % (layer.name, 1, 1))
        layer_param = pp + layer_param

        ncnn_net.append(' '.join(layer_param))

        for w in layer.weights:
            ncnn_weights.append(w)

    def convert(self):
        ncnn_net = []
        ncnn_weights = []

        layers = self._get_layers()

        for layer in layers:
            layer_ncnn_lst = self._dump_layer_weights(layer)
            if layer_ncnn_lst:
                for layer_ncnn in layer_ncnn_lst:
                        self.link_ncnn(layer_ncnn, ncnn_net, ncnn_weights)

        text_net = '\n'.join(ncnn_net)
        text_net = ('%d %d\n' % (len(ncnn_net), len(ncnn_net))) + text_net
        text_net = '7767517\n' + text_net

        return text_net, ncnn_weights

if __name__ == '__main__':
    ''' Initial Logger '''
    logger = initial_logger()

    ''' Parse Command '''
    model_name = parseCommandLine(sys.argv[1:])

    ''' Initial Converter '''
    k2n_converter = Converter(model_name, logger)

    ''' DO Conversion '''
    text_net, binary_weights = k2n_converter.convert()

    ''' Save files '''
    ModelDir = '.\\'
    NetName = 'model_NCNN'
    with open(ModelDir + NetName + '.param', 'w') as f:
        f.write(text_net)
    with open(ModelDir + NetName + '.bin', 'w') as f:
        for weights in binary_weights:
            for blob in weights:
                blob_32f = blob.flatten().astype(np.float32)
                blob_32f.tofile(f)
