from math import ceil
from typing import List
from gym.spaces import Space, Discrete, Box, Dict

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers.dense_attention import BaseDenseAttention
from misc.utils import optional_device, kwargify

Layer = layers.Layer

def FeedforwardModel(
        n_layers: int,
        n_nodes: int,
        n_actions: int,
        normalize: True):
    seq_layers = []
    for i in range(n_layers):
        seq_layers += [layers.Dense(n_nodes), layers.ReLU()]
        if normalize:
            seq_layers += [layers.LayerNormalization()]
    seq_layers += [layers.Dense(n_actions)]
    return keras.Sequential(seq_layers)

class AttentionLayer(BaseDenseAttention):
    """
        should be called module((q, k, v))
        NOTE: not sure if this works if query_dim != value_dim
        :param query: tf.Tensor[f32, dev] : (batch, n_queries, query_dim)
        :param key: tf.Tensor[f32, dev] : (batch, n_values, query_dim)
        :param value: tf.Tensor[f32, dev] : (batch, n_values, value_dim)
        :returns info: tf.Tensor[f32, dev] : (batch, n_queries, value_dim)
    """
    def __init__(self,
            use_scale: bool = False,
            device: tf.device = None,
            **kwargs):
        super().__init__()
        self.device = device
        self.use_scale: bool = use_scale

    def build(self, input_shape):
        super().build(input_shape)


    def _calculate_scores(self,
            query: tf.Tensor,
            key: tf.Tensor):
        """
        :param query: tf.Tensor[f32, dev] : (batch, n_query, query_dims)
        :param key: tf.Tensor[f32, dev] : (batch, n_key, query_dims)
        :return:tf.Tensor[f32, dev] (batch, n_query, n_key)
        """
        with self.device:
            scores = tf.matmul(query, key, transpose_b=True)
            # (batch, n_query, n_key)
            if self.use_scale:
                key_dims: int = key.shape[-1]
                norm_factor = tf.sqrt(tf.cast(key_dims, tf.float32))
                scores /= norm_factor
        return scores

class ProjectedAttentionLayer(Layer):
    def __init__(self,
                 encoding_dim: int,
                 use_scale: bool = True,
                 device: tf.device = None,
                 **kwargs):
        """

        :param encoding_dim:  number of dimensions to linearly encode key, value and query in
            if -1, will simply use key query and value as provided (e.g. identity encoder)
        :param use_scale: whether or not to scale scores by 1/sqrt(d)
        :param device: device to place everything ong
        :param kwargs: comaptibility
        """
        super().__init__()
        device: tf.device = optional_device(device)
        self.encode: bool = encoding_dim != -1
        self.encoding_dim: int = encoding_dim
        self.device: tf.device = optional_device(device)
        self.use_scale: bool = use_scale

        if self.encode:
            with self.device:
                self.key_encoder: Layer = layers.Dense(encoding_dim)
                self.query_encoder: Layer = layers.Dense(encoding_dim)
                self.value_encoder: Layer = layers.Dense(encoding_dim)

    def _calculate_scores(self,
            queries: tf.Tensor,
            keys: tf.Tensor):
        """
        :param queries: tf.Tensor[f32, dev] : (batch, n_query, query_dims)
        :param keys: tf.Tensor[f32, dev] : (batch, n_key, query_dims)
        :return:tf.Tensor[f32, dev] (batch, n_query, n_key)
        """
        with self.device:
            if self.encode:
                queries: tf.Tensor = self.query_encoder(queries)
                # (batch, n_query, n_dims)
                keys: tf.Tensor = self.key_encoder(keys)
                # (batch, n_key, n_dims)
            scores = tf.matmul(queries, keys, transpose_b=True)
            # (batch, n_query, n_key)
            if self.use_scale:
                key_dims: int = keys.shape[-1]
                norm_factor = tf.sqrt(tf.cast(key_dims, tf.float32))
                scores /= norm_factor
            return scores

    def alignment(self,
            queries: tf.Tensor,
            keys: tf.Tensor) -> tf.Tensor:
        """
        :param queries: tf.Tensor[f32, dev] : (batch, n_query, query_dims)
        :param keys: tf.Tensor[f32, dev] : (batch, n_key, query_dims)
        :return:tf.Tensor[f32, dev] (batch, n_query, n_key)
            last dim sums to one
        """
        with self.device:
            scores: tf.Tensor = self._calculate_scores(queries, keys)
            # tf.Tensor[f32, dev] (batch, n_query, n_key)
            return tf.nn.softmax(scores, axis=2)

    def build(self, input_shape):
        super().build(input_shape)

        if not self.encode:
            query_dims: int = input_shape[0][-1]
            key_dims: int = input_shape[1][-1]
            val_dims: int = input_shape[2][-1]
            assert query_dims == key_dims
            self.encoding_dim = val_dims
        else:
            # build submodules
            self.query_encoder.build(input_shape[0])
            self.key_encoder.build(input_shape[1])
            self.value_encoder.build(input_shape[2])

    def call(self, qkv):
        """
        :param qkv: tuple of queries, keys and values
          -param queries: tf.Tensor[f32, dev] : (batch, n_query, query_dims)
          -param keys: tf.Tensor[f32, dev] : (batch, n_key, key_dims)
          -param values: tf.Tensor[f32, dev] : (batch, n_key, value_dims)
        :return:tf.Tensor[f32, dev] : (batch, n_query, n_dims)
        """
        queries, keys, values = qkv
        with self.device:
            alignments: tf.Tensor = self.alignment(queries, keys)
            # tf.Tensor[f32, dev] : (batch, n_query, n_key)
            if self.encode:
                values = self.value_encoder(values)
                # tf.Tensor[f32, dev] : (batch, n_query, n_dims)
            return alignments @ values

class MultiHeadAttentionLayer(layers.Layer):
    def __init__(self,
            n_heads: int,
            encoding_dim: int,
            device: tf.device = None,
            **kwargs):
        kwargs = kwargify(locals())
        super().__init__()
        self.n_heads: int = n_heads
        self.encoding_dim: int = encoding_dim
        self.device: tf.device = optional_device(device)

        self.attentions = [ProjectedAttentionLayer(**kwargs) for _ in range(n_heads)]

    def call(self, qkv):
        """

        :param qkv: tuple of queries, keys, values
            -param queries: tf.Tensor[f32, dev] : (batch, n_query, query_dims)
          -param keys: tf.Tensor[f32, dev] : (batch, n_key, key_dims)
          -param values: tf.Tensor[f32, dev] : (batch, n_key, value_dims)
        :return: (batch, n_query, encoding_dims * n_heads)
        """
        result: List[tf.Tensor] = []
        for attn in self.attentions:
            result.append(attn(qkv))
        return tf.concat(result, axis=2)

class TransformerBaseLayer(layers.Layer):
    def __init__(self,
            n_heads: int,
            hidden_nodes: int,
            device: tf.device = None,
            **kwargs):
        super().__init__()
        self.n_heads: int = n_heads
        self.hidden_nodes: int = hidden_nodes
        self.kwargs = kwargs
        self.device = optional_device(device)

        self.mh_attn: MultiHeadAttentionLayer = ...
        self.ffnn: keras.Sequential = ...

    def build(self, input_shape):
        n_tokens, n_dims = input_shape[1:]
        encoding_dim = int(ceil(n_dims / self.n_heads))

        with self.device:
            self.mh_attn = \
                MultiHeadAttentionLayer(self.n_heads, encoding_dim, self.device, **self.kwargs)
            self.ffnn = keras.Sequential([
                layers.Dense(self.hidden_nodes),
                layers.ReLU(),
                layers.Dense(n_dims)])

        super().build(input_shape)

class TransformerLayer(TransformerBaseLayer):
    def __init__(self,
            n_heads: int,
            hidden_nodes: int,
            device: tf.device = None,
            bypass: bool = False,
            **kwargs):
        super().__init__(n_heads, hidden_nodes, device, **kwargs)
        self.bypass: bool = bypass

    def call(self, values, **kwargs):
        attn_out = self.mh_attn((values, ) * 3)
        if self.bypass:
            attn_out += values
        lin_out = self.ffnn(attn_out)
        if self.bypass:
            lin_out += attn_out
        return lin_out

class TrXL_Layer(TransformerBaseLayer):
    def __init__(self,
            n_heads: int,
            hidden_nodes: int,
            device: tf.device = None,
            **kwargs):
        super().__init__(n_heads, hidden_nodes, device, **kwargs)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, values, **kwargs):
        """

        :param values: tf.Tensor[f32, dev] : (batch, n_values, value_dim)
        :return: tf.Tensor[f32, dev]: (batch, n_values, value_dim)
        """
        # shape is the same throughout
        attn_out = self.mh_attn((values, values, values))
        attn_out += values
        norm_out = self.norm1(attn_out)
        lin_out = self.ffnn(norm_out)
        lin_out += norm_out
        return self.norm2(lin_out)

class TrXLI_Layer(TransformerBaseLayer):
    def __init__(self,
            n_heads: int,
            hidden_nodes: int,
            device: tf.device = None,
            **kwargs):
        super().__init__(n_heads, hidden_nodes, device, **kwargs)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, values, **kwargs):
        """
        :param values: tf.Tensor[f32, dev] : (batch, n_values, value_dim)
        :return: tf.Tensor[f32, dev]: (batch, n_values, value_dim)
        """
        # shape is the same throughout
        norm1_out = self.norm1(values)
        attn_out = self.mh_attn((norm1_out, ) * 3)
        attn_out += values
        norm2_out = self.norm2(attn_out)
        lin_out = self.ffnn(norm2_out)
        lin_out += attn_out
        return lin_out

class TransformerModel(keras.Model):
    def __init__(self,
            n_layers: int,
            n_heads: int,
            hidden_nodes: int,
            encoding_dim: int,
            device: tf.device = None,
            layer_type: str = 'classic',
            **kwargs):
        super().__init__()
        assert hidden_nodes % n_heads == 0
        # output of an MHA layer is n_heads * ceil(hidden_nodes/ n_heads)
        # because we have bypass layers, we need the input and output to be the same
        self.device = optional_device(device)

        self.encoder = MultiHeadAttentionLayer(n_heads, encoding_dim, device, **kwargs)
        self.initial_projection = layers.Dense(hidden_nodes)

        layer_type = layer_type.lower()
        if layer_type == 'transformer':
            self.attn_layers: List[Layer] = [TransformerLayer(n_heads, hidden_nodes, device, **kwargs)
                for _ in range(n_layers)]
        elif layer_type == 'trxl':
            self.attn_layers: List[Layer] = [TrXL_Layer(n_heads, hidden_nodes, device, **kwargs)
                for _ in range(n_layers)]
        elif layer_type == 'trxli':
            self.attn_layers: List[Layer] = [TrXLI_Layer(n_heads, hidden_nodes, device, **kwargs)
                for _ in range(n_layers)]
        else:
            raise Exception(f"Expected layer_type in [classic, trxl, trxli], got {layer_type}")

        self.out_layer = layers.Dense(1)

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """
        :param station: should contain the following in a tuple
            :param stations: (batch_dim, n_stations, station_dims)
            :param cars: (batch_dim, n_cars, car_dim)
        :return: (batch_dim, n_stations)
        """
        stations, cars = data
        with self.device:
            car_info: tf.Tensor = self.encoder((stations, cars, cars))
            # (batch_dim, n_stations, encoding_dim)
            data: tf.Tensor = tf.concat((stations, car_info), axis=2)
            # (batch_dim, n_stations, station_dims + encoding_dims)
            data = self.initial_projection(data)
            # (batch_dim, n_stations, hidden_nodes)
            for attn in self.attn_layers:
                data = attn(data)
                # same size
            out = self.out_layer(data)
            # (batch_dim, n_stations, 1)
            out = tf.squeeze(out, axis=2)
            # (batch_dim, n_stations)
            return out


def make_model(
        action_space: Space,
        observation_space: Space,
        model: str,
        device: tf.device,
        n_nodes: int,
        n_layers: int,
        normalize: bool = True,
        n_heads: int = 1,
        **kwargs) -> keras.Model:
    """

    :param action_space: Space (Discrete)
        Used to calculate the shape of model output
    :param observation_space: Space (see model)
        Used to calculate the shape of model input
    :param model: str
        one of [feedforward, transformer]
        Type of model to make
    :param device: tf.device
        device to put the model on
    :param n_nodes: int
        number of nodes per layer (or per head)
    :param n_layers: int
        number of layers
    :param normalize: bool
        whether or not to use LayerNorm
    :param n_heads: int
        number of heads to use (if model uses attention)
    :param kwargs:
        for compatibility
    :return: nn.Module
        see docs for individual modules for input
    """
    action_space: Space = action_space
    observation_space: Space = observation_space
    assert isinstance(action_space, Discrete)
    n_actions: int = action_space.n
    model: str = model.lower()
    if model == 'feedforward':
        assert isinstance(observation_space, Box)
        return FeedforwardModel(n_layers, n_nodes, n_actions, **kwargs)
    elif model in ['trxl', 'trxli', 'transformer']:
        assert isinstance(observation_space, Dict)
        assert 'cars' in observation_space.spaces and 'stations' in observation_space.spaces
        car_dims: int = observation_space['cars'].shape[1]
        station_dims: int = observation_space['stations'].shape[1]
        assert n_nodes % n_heads == 0
        encoding_dim = n_nodes / n_heads
        return TransformerModel(n_layers, n_heads, hidden_nodes=n_nodes,
                encoding_dim=encoding_dim, layer_type=model, **kwargs)
    else:
        raise Exception(f"model must be in [feedforward, transformer], got {model}")

