# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ricga.reference.tf2keras_image_process import tf2keras_image


def distort_image(image, thread_id):
    """Perform random distortions on an image.
  
    Args:
      image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions. There should be a multiple of 2 preprocessing threads.
  
    Returns:
      distorted_image: A float32 Tensor of shape [height, width, 3] with values in
        [0, 1].
    """
    # Randomly flip horizontally.
    with tf.name_scope("flip_horizontal", values=[image]):
        image = tf.image.random_flip_left_right(image)

    # Randomly distort the colors based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  ssd_model,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format="jpeg"):
    """Decode an image, resize and apply random distortions.
  
    In training, images are distorted slightly differently depending on thread_id.
  
    Args:
      encoded_image: String Tensor containing the image.
      is_training: Boolean; whether preprocessing for training or eval.
      height: Height of the output image.
      width: Width of the output image.
      ssd_model: SSD300 model.
      resize_height: If > 0, resize height before crop to final dimensions.
      resize_width: If > 0, resize width before crop to final dimensions.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions. There should be a multiple of 2 preprocessing threads.
      image_format: "jpeg" or "png".
  
    Returns:
      A float32 Tensor of shape [height, width, 3] with values in [-1, 1].
  
    Raises:
      ValueError: If image_format is invalid.
    """

    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in half of the thread.
    def image_summary(name, image_to_sum):
        if thread_id % 2 == 0:
            tf.summary.image(name, tf.expand_dims(image_to_sum, 0))

    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == "jpeg":
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    original_image = image

    image_summary("original_image", image)

    # Resize image.
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        image = tf.image.resize_images(image,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)

    # Crop to final dimensions.
    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        # Central crop, assuming resize_height > height, resize_width > width.
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    image_summary("resized_image", image)

    # Randomly distort the image.
    if is_training:
        image = distort_image(image, thread_id)

    image_summary("final_image", image)

    # Rescale to [-1,1] instead of [0, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    # ssd process
    image_300x300 = tf.image.resize_images(original_image, [300, 300])
    image_300x300_ssd_input = tf2keras_image(image_300x300)

    # with tf.variable_scope("ssd"):
    ssd_output = ssd_model(tf.expand_dims(image_300x300_ssd_input, 0))[0]

    with tf.variable_scope("ssd_out_processing"):
        mbox_loc = ssd_output[:, :4]
        variances = ssd_output[:, -4:]
        mbox_priorbox = ssd_output[:, -8:-4]
        mbox_conf = ssd_output[:, 4:-8]
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = tf.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = tf.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = tf.expand_dims(decode_bbox_center_x - 0.5 * decode_bbox_width, -1)
        decode_bbox_ymin = tf.expand_dims(decode_bbox_center_y - 0.5 * decode_bbox_height, -1)
        decode_bbox_xmax = tf.expand_dims(decode_bbox_center_x + 0.5 * decode_bbox_width, -1)
        decode_bbox_ymax = tf.expand_dims(decode_bbox_center_y + 0.5 * decode_bbox_height, -1)
        decode_bbox = tf.concat((decode_bbox_ymin,
                                 decode_bbox_xmax,
                                 decode_bbox_ymax,
                                 decode_bbox_xmin), axis=-1)
        decode_bbox = tf.minimum(tf.maximum(decode_bbox, 0.0), 1.0)

        mbox_conf_without_background = tf.slice(mbox_conf, [0, 1], [-1, -1])
        mbox_conf_max = tf.reduce_max(mbox_conf_without_background, 1)
        idx = tf.image.non_max_suppression(decode_bbox, mbox_conf_max, max_output_size=1)
        idx = tf.reshape(idx, [1])
        good_box = decode_bbox[idx[0]]

    region_image = tf.image.crop_and_resize(tf.expand_dims(image_300x300, 0),
                                            boxes=tf.expand_dims(good_box, 0),
                                            box_ind=tf.constant([0], dtype=tf.int32),
                                            crop_size=[height, width],
                                            name="region_images")[0]
    image_summary("region_image", region_image)
    # Rescale to [-1,1] instead of [0, 1]
    region_image = tf.subtract(region_image, 0.5)
    region_image = tf.multiply(region_image, 2.0)

    return image, region_image
    # return ssd, region_image
