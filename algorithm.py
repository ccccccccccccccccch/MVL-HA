import tensorflow as tf
from keras import backend as K
from tensorflow import keras


class IntraAttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        # self.supports_masking = True
        super(IntraAttentionLayer, self).__init__(**kwargs)
        # self.name = "intra"

    def build(self, input_shape):  # (  ,64)
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True,
            name='w'
        )
        super(IntraAttentionLayer, self).build(input_shape)

    def call(self, x):  # x(32,64)
        x = K.expand_dims(x, axis=1)  # (32,1,64)
        attention = K.softmax(K.dot(x, self.W))  # (32,1,64) *(64,64)  = (32,1,64)
        final = K.squeeze(x * attention, axis=1)  # (32,1,64)  -> (32,64)
        return final, attention  # (32,64)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    #   return input_shape[0], input_shape[-1]  #

    def get_config(self):
        config = super().get_config().copy()
        # config.update({
        #    'supports_masking': self.supports_masking
        # })
        return config


class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        # self.name = "concat"

    # input_shape: [TensorShape([None， 30]),TensorShape([None, 30])，TensorShape([None， 30]))
    def build(self, input_shape):  # [(32,64) (32,64) (32,64) (32,64) (32,64) (32,64)]
        print('----------------------------------input_shape', input_shape)
        self.W = self.add_weight(
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True,
            name='W'
        )
        self.w = self.add_weight(
            shape=(input_shape[0][-1], 1),
            initializer="random_normal",
            trainable=True,
            name='w'
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):  # [6x(32,64)]
        inputs = tf.stack(x, axis=1)  # (32,6,64)
        v = K.tanh(K.dot(inputs, self.W))  # (32,6,64) * (64,64)  = (32,6,64)
        v1 = K.dot(v, self.w)  # (32,6,64) * (64,1)  = (32,6,1)
        attention = K.softmax(v1, axis=1)  # (32,6,1)
        result = K.sum(inputs * attention, axis=1)  # (32,6,64)  -> (32,64)
        print('call:', result)
        return result, attention

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0])

    def get_config(self):
        config = super().get_config().copy()
        return config


def multiview_han(finance_unit, stock_unit, network_unit, attention_unit, mlp_unit,
                  finance_drop, stock_drop, network_drop):
    finance_input = keras.Input(shape=(22,), name="finance_input")
    stock_input = keras.Input(shape=(244, 8), name="stock_input")
    network_input = keras.Input(shape=(4,), name="network_input")

    stock_masking = keras.layers.Masking(mask_value=0)(stock_input)

    finance_feature = keras.layers.Dense(units=finance_unit)(finance_input)
    finance_feature = keras.layers.BatchNormalization()(finance_feature)
    finance_feature = keras.layers.Activation(activation="relu")(finance_feature)
    finance_feature = keras.layers.Dropout(rate=finance_drop)(finance_feature)

    stock_feature = keras.layers.LSTM(units=stock_unit)(stock_masking)
    # stock_feature = keras.layers.BatchNormalization()(stock_feature)
    stock_feature = keras.layers.Dropout(rate=stock_drop)(stock_feature)

    network_feature = keras.layers.Dense(units=network_unit)(network_input)
    network_feature = keras.layers.BatchNormalization()(network_feature)
    network_feature = keras.layers.Activation(activation='relu')(network_feature)
    network_feature = keras.layers.Dropout(rate=network_drop)(network_feature)

    finance_feature = keras.layers.Dense(units=attention_unit)(finance_feature)
    stock_feature = keras.layers.Dense(units=attention_unit)(stock_feature)
    network_feature = keras.layers.Dense(units=attention_unit)(network_feature)

    # IntraAttention
    finance_ah, _ = IntraAttentionLayer()(finance_feature)
    stock_ah, _ = IntraAttentionLayer()(stock_feature)
    network_ah, _ = IntraAttentionLayer()(network_feature)

    # InterAttention
    x, _ = AttentionLayer()([finance_ah, stock_ah, network_ah])

    x = keras.layers.Dense(units=mlp_unit)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation="relu")(x)

    pre_out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=[finance_input, stock_input, network_input],
                        outputs=pre_out)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")],
                  )

    return model
