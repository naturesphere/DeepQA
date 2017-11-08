# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
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
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf
import numpy as np
import copy
from chatbot.textdata import Batch
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    '''
    def diverse_embedding_rnn_decoder(self,
        decoder_inputs,
        initial_state,
        cell,
        num_symbols,
        embedding_size,
        output_projection=None,
        feed_previous=False,
        update_embedding_for_previous=True,
        scope=None,
        k=3 ):
        with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
            if output_projection is not None:
                dtype = scope.dtype
                proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
                proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
                proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
                proj_biases.get_shape().assert_is_compatible_with([num_symbols])

            embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
            def _extract_one_of_topk_and_embed(embedding, k=3,
                              output_projection=None,
                              update_embedding=True
                              ):
                def loop_function(prev, _):
                    if output_projection is not None:
                        prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
                    # _, prev_symbols = nn_ops.top_k(prev, k) # prev_symbols.shape = batchsize * k
                    # idx = np.random.randint(k)
                    # print("idx = "+ str(idx))
                    # prev_symbol = prev_symbols[:,idx]

                    prev_symbol = math_ops.argmax(prev, 1)
                    # prev_symbol = prev[:,1]
                    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
                    if not update_embedding:
                        emb_prev = array_ops.stop_gradient(emb_prev)
                    return emb_prev
                return loop_function

            loop_function = _extract_one_of_topk_and_embed( embedding, k, output_projection,
                update_embedding_for_previous) if feed_previous else None
            emb_inp = (embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
            return tf.contrib.legacy_seq2seq.rnn_decoder(emb_inp, initial_state, cell, loop_function=loop_function)

    def diverse_embedding_rnn_seq2seq(self,
        encoder_inputs,
        decoder_inputs,
        cell,
        num_encoder_symbols,
        num_decoder_symbols,
        embedding_size,
        output_projection=None,
        feed_previous=False,
        dtype=None,
        scope=None,
        k=3):
        with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
            if dtype is not None:
                scope.set_dtype(dtype)
            else:
                dtype = scope.dtype
        print("diverse_embedding_rnn_seq2seq")
        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            encoder_cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        # if isinstance(feed_previous, bool):
        return self.diverse_embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            k=k)
    '''

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt     = tf.cast(outputProjection.W_t,             tf.float32)
                localB      = tf.cast(outputProjection.b,               tf.float32)
                localInputs = tf.cast(inputs,                           tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        self.textData.getVocabularySize()),  # The number of classes
                    self.dtype)

        # Creation of the rnn cell
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                self.args.hiddenSize,
            )
            if not self.args.test:  # TODO: Should use a placeholder instead
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation

        # decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
        #     self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
        #     self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
        #     encoDecoCell,
        #     self.textData.getVocabularySize(),
        #     self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
        #     embedding_size=self.args.embeddingSize,  # Dimension of each word
        #     output_projection=outputProjection.getWeights() if outputProjection else None,
        #     feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        # )

        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.textData.getVocabularySize(),
            self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embeddingSize,  # Dimension of each word
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # print("before self.diverse_embedding_rnn_seq2seq")
        # decoderOutputs, states = self.diverse_embedding_rnn_seq2seq(
        #     self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
        #     self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
        #     encoDecoCell,
        #     self.textData.getVocabularySize(),
        #     self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
        #     embedding_size=self.args.embeddingSize,  # Dimension of each word
        #     output_projection=outputProjection.getWeights() if outputProjection else None,
        #     feed_previous=bool(self.args.test),  # When we test (self.args.test), we use previous output as next input (feed_previous)
        #     k=3
        # )

        # print("buildNetwork: self.textData.getVocabularySize {}".format(self.textData.getVocabularySize()))
        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        if self.args.test:
            if not outputProjection:
                # print("test is decoderOutputs")
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

            # TODO: Attach a summary to visualize the output

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
