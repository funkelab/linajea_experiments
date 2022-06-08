import os

import tensorflow as tf


class ResNet(tf.keras.Model):

    def __init__(
        self,
        **kwargs
    ):
        super(ResNet, self).__init__(name="ResNet", **kwargs)
        self.resnetbase = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_tensor=None,
            input_shape=None, pooling='avg')
        self.out = tf.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same")

    def call(self, inputs):
        x = self.resnetbase(inputs)
        logits = self.out(x)

        pred_labels = tf.argmax(input=logits, axis=1)
        pred_probs = tf.nn.softmaX(logits)

        return pred_labels, pred_probs


def mk_net(config, output_folder):

    model = ResNet()
    fn = os.path.join(output_folder,
                      config.model.net_name + "_checkpoint_0")
    model.save(fn)
