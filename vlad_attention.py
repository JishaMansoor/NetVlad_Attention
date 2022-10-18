from tensorflow import flags
import tensorflow as tf
import frame_level_models
import video_level_models
import models
import model_utils as utils
import tensorflow.contrib.slim as slim
from tensorflow import  matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.backend import softmax
import math
FLAGS = flags.FLAGS


# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        dim=  cast(d_k, tf.float32)
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(dim)
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
 
# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k,activation = None)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k,activation = None)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v,activation = None)  # Learned projection matrix for the values
        self.W_o = Dense(d_model,activation = None)  # Learned projection matrix for the multi-head output
        self.depth = self.d_model // self.heads
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            print("flag " , shape(x)[0])
            print("flag " , shape(x)[1])
            x = reshape(x, shape=(shape(x)[0], -1, heads, self.depth))
            #x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            print("not flag",x)
            print("self.dk",self.d_k)
            x = reshape(x, shape=(shape(x)[0], -1,self.d_model))
        return x
 
    def call(self, queries,keys,values, mask=None):
        print("call query ",queries)
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        print("o_reshaped", o_reshaped)
        print("self.head", self.heads)
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        normalized_output = tf.nn.l2_normalize(output, 1)
        normalized_output = tf.reshape(normalized_output, [-1, self.d_model])
        output = tf.nn.l2_normalize(normalized_output)
        return self.W_o(output)

class NetVLADAttnModel(models.BaseModel):
  """Creates a NetVLAD based model with attention.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    print("cluster size ", cluster_size)


    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)


    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    print("feature size ", feature_size)
    video_NetVLAD = frame_level_models.NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = frame_level_models.NetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)


    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])
    early_attention =  False
    if early_attention:
         video_features = vlad_video #tf.reshape(vlad_video, [-1, max_frames, 1024])
         video_features = tf.reshape(vlad_video,[-1 ,cluster_size ])#tf.reshape(vlad_video, [-1, max_frames, 1024])
         print("jisha ",vlad_video.get_shape())
         audio_features = vlad_audio #tf.reshape(vlad_audio, [-1, max_frames, 128])
         audio_features = tf.reshape(vlad_audio,[-1,cluster_size//2]) #tf.reshape(vlad_audio, [-1, max_frames, 128])
         print( "video_feature " , video_features.get_shape()[1])
         h=1
         video_attn_layer =  MultiHeadAttention(h, d_k=video_features.get_shape()[1],d_v=video_features.get_shape()[1], d_model=video_features.get_shape()[1])
         audio_attn_layer =  MultiHeadAttention(h, d_k=audio_features.get_shape()[1], d_v=audio_features.get_shape()[1],d_model=audio_features.get_shape()[1])

         vlad_video_attn=video_attn_layer(video_features,video_features,video_features)
         vlad_audio_attn=audio_attn_layer(audio_features,audio_features,audio_features)
         vlad_video=vlad_video_attn
         vlad_video=tf.reshape(vlad_video_attn,[-1,cluster_size*1024])
         vlad_audio=vlad_audio_attn
         vlad_audio=tf.reshape(vlad_audio_attn,[-1,(cluster_size//2) * 128])

    print("vlad_video after attn", vlad_video)
    print("vlad_audio after attn", vlad_audio)

    vlad = tf.concat([vlad_video, vlad_audio],1)
    print("vlad after concat ",vlad)
    vlad_dim = vlad.get_shape().as_list()[1]
    print("vlad_dim ",vlad_dim)

    cs=cluster_size
    hidden1_weights = tf.get_variable("hidden1_weights", [vlad_dim, hidden1_size],initializer=tf.random_normal_initializer(stddev=1/math.sqrt(cs)))

    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)


    if (gating and not early_attention):
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

        gates = tf.matmul(activation, gating_weights)

        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)


        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)  
