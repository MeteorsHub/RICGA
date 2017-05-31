import keras.backend as K
import tensorflow as tf

from ricga.reference.ssd import SSD300

sess = tf.Session()
sess.run(tf.global_variables_initializer())
K.set_session(sess)
with tf.variable_scope("SSD300"):
    model = SSD300((300, 300, 3))
    model.load_weights("/home/meteorshub/code/ricga/ricga/data/weights_SSD300.hdf5", by_name=True)

SSD300_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="SSD300")
saver = tf.train.Saver(SSD300_variables, write_version=tf.train.SaverDef.V1)
# saver.save(sess, "temp/ssd300.ckpt", global_step=0)
saver.restore(sess, "temp/ssd300.ckpt")
SSD300_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="SSD300")
print('test')
