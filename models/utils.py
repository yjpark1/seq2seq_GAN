import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.seq2seq.python.ops.helper import Helper
import tensorflow_probability as tfp


def dynamic_time_pad(recon_output, max_len, batch_size):
    shape = recon_output.get_shape().as_list()
    shape_op = tf.shape(recon_output)

    pad_size = max_len - shape_op[1]
    pad_type = recon_output.dtype

    pad_tensor = tf.zeros([batch_size, pad_size, shape[2]], dtype=pad_type)

    recon_output = tf.concat([recon_output, pad_tensor], axis=1)
    return recon_output


class CustomGreedyEmbeddingHelper(Helper):
    """A helper for use during inference.
    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token):
        """Initializer.
        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`. The returned tensor
            will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
            scalar.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            # self._embedding_fn = (
            #     lambda ids: embedding_ops.embedding_lookup(embedding, ids))
            print('error!')

        self._start_tokens = start_tokens
        self._end_token = end_token
        self._batch_size = tf.shape(self._start_tokens)[0]
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return finished, self._start_inputs

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time  # unused by next_inputs_fn
        # gumbel trick
        # http://anotherdatum.com/gumbel-gan.html
        dist = tfp.distributions.RelaxedOneHotCategorical(1e-2, probs=outputs)
        outputs = dist.sample()

        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(outputs))
        return finished, next_inputs, state

