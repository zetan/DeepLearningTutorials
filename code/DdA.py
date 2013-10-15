"""
 a three layer denoising auto-encoder

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA

from utils import tile_raster_images
from theano import function
import PIL.Image

class SdA(object):
    

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500]):
      
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
    


        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)
       
       
    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.3),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    

   
def test_SdA(pretraining_epochs=20,
             pretrain_lr=0.1,
             dataset='../data/mnist.pkl.gz', batch_size=20, output_folder='DdA_plots_test'):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[400, 225, 100])
    size_sqrt = [28,20,15,10]  # input included
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.3, .2, .3]
    

    x = theano.tensor.dmatrix()
    y = theano.tensor.dmatrix()
    z = theano.tensor.dmatrix()
    getW12 = theano.function([x,y], T.dot(x,y))
    getW123 = theano.function([x,y,z], T.dot(T.dot(x,y),z)) 
    
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

            
            if i == 0:
                image = PIL.Image.fromarray(
                    tile_raster_images(X=sda.dA_layers[i].W.get_value(borrow=True).T,
                           img_shape=(size_sqrt[i], size_sqrt[i]), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
                image.save('Pre-training layer %i, epoch %d.png'  % (i, epoch))
            if i == 1:
                w = getW12(sda.dA_layers[1].W.get_value(borrow=True).T, sda.dA_layers[0].W.get_value(borrow=True).T)
                image = PIL.Image.fromarray(
                    tile_raster_images(X=w,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
                image.save('Pre-training layer %i, epoch %d.png'  % (i, epoch))
            
            if i == 2:
                w = getW123(sda.dA_layers[2].W.get_value(borrow=True).T, sda.dA_layers[1].W.get_value(borrow=True).T, sda.dA_layers[0].W.get_value(borrow=True).T)
                image = PIL.Image.fromarray(
                    tile_raster_images(X=w,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
                image.save('Pre-training layer %i, epoch %d.png'  % (i, epoch))
           
    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

   

if __name__ == '__main__':
    test_SdA()
