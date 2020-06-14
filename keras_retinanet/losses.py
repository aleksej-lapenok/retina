"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import tensorflow as tf
# slim = tf.contrib.slim
from . import backend


# class MultiLossLayer():
#     def __init__(self, loss_list):
#         self._loss_list = loss_list
#         self._sigmas_sq = []
#         for i in range(len(self._loss_list)):
#             self._sigmas_sq.append(slim.variable('Sigma_sq_' + str(i), dtype=tf.float32, shape=[], initializer=tf.initializers.random_uniform(minval=0.2, maxval=1)))
#
#     def get_loss(self):
#         factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[0]))
#         loss = tf.add(tf.multiply(factor, self._loss_list[0]), tf.log(self._sigmas_sq[0]))
#         for i in range(1, len(self._sigmas_sq)):
#             factor = tf.div(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
#             loss = tf.add(loss, tf.add(tf.multiply(factor, self._loss_list[i]), tf.log(self._sigmas_sq[i])))
#         return loss

def focal(alpha=0.25, gamma=2.0, sigma_var=None):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    if sigma_var is None:
        sigma_var = tf.Variable(dtype=tf.float32, name="sigma_sq_focal",
                                initial_value=tf.random_uniform_initializer(minval=-2, maxval=5)
                                .__call__(shape=[], dtype=tf.float32),
                                trainable=True)

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification * keras.backend.exp(-sigma_var),
                                     classification * keras.backend.exp(-sigma_var))
        focal_weight = alpha_factor * focal_weight ** gamma

        cross_entropy = keras.backend.binary_crossentropy(labels, classification) * keras.backend.exp(-sigma_var) + sigma_var / 2

        cls_loss = focal_weight * cross_entropy

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer
        # return MultiLoss([keras.backend.sum(cls_loss) / normalizer]).get_loss()

    # multiLoss = MultiLoss([_focal])
    # return multiLoss.get_loss, multiLoss
    return _focal, sigma_var


def smooth_l1(sigma=3.0, sigma_var=None):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    if sigma_var is None:
        sigma_var = tf.Variable(dtype=tf.float32, name="sigma_sq_smoth_l1",
                                initial_value=tf.random_uniform_initializer(minval=-2, maxval=5)
                                .__call__(shape=[], dtype=tf.float32),
                                trainable=True)

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = backend.where(keras.backend.equal(anchor_state, 1))
        regression = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

        factor = 1.0 / (2.0 * keras.backend.exp(sigma_var))
        return factor * (keras.backend.sum(regression_loss) / normalizer) + 0.5 * sigma_var
        # return MultiLoss([keras.backend.sum(regression_loss) / normalizer]).get_loss()

    # loss_class = MultiLoss([_smooth_l1])
    # return loss_class.get_loss, loss_class
    return _smooth_l1, sigma_var
