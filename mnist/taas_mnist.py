# coding=utf-8

from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from caicloud.clever.tensorflow import dist_base

tf.app.flags.DEFINE_string("data_dir",
                           "/tmp/mnist-data",
                           "path to mnist dataset.")
tf.app.flags.DEFINE_string("checkpoint_dir",
                           None,
                           "path to pre-trained model checkpoint.")

FLAGS = tf.app.flags.FLAGS

# 从指定目录中读取 MNIST 训练、验证和测试数据集。
_mnist = read_data_sets(FLAGS.data_dir, one_hot=True)

_input_images = None
_labels = None
_global_step = None
_loss = None
_train_op = None

def model_fn(sync, num_replicas):
    global _input_images, _loss, _labels, _train_op
    global _mnist, _global_step

    # 构建推理模型
    _input_images = tf.placeholder(tf.float32, [None, 784], name='image')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')
    logits = tf.matmul(_input_images, W) + b

    _global_step = tf.Variable(0, name='global_step', trainable=False)

    # 定义模型损失和优化器
    _labels = tf.placeholder(tf.float32, [None, 10], name='labels')
    _loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_labels),
        name='loss')
    optimizer = tf.train.AdagradOptimizer(0.01);
    _train_op = optimizer.minimize(_loss, global_step=_global_step)

    return dist_base.ModelFnHandler(
        global_step=_global_step)

_local_step = 0
def train_fn(session, num_global_step):
    global _local_step, _train_op

    start_time = time.time()
    _local_step += 1
    batch_xs, batch_ys = _mnist.train.next_batch(100)
    feed_dict = {_input_images: batch_xs,
                 _labels: batch_ys}
    _, loss_value, np_global_step = session.run(
        [_train_op, _loss, _global_step],
        feed_dict=feed_dict)
    duration = time.time() - start_time
    
    if _local_step % 50 == 0:
        print('Step {0}: loss = {1:0.2f} ({2:0.3f} sec), global step: {3}.'.format(
            _local_step, loss_value, duration, np_global_step))

def gen_init_fn():
    checkpoint_path = FLAGS.checkpoint_dir
    if checkpoint_path is None or checkpoint_path == "":
        return None
    
    if not tf.gfile.Exists(checkpoint_path):
        print('WARNING: checkpoint path {0} not exists.'.format(checkpoint_path))
        return None
    
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    # 定义 tf.train.Saver 会修改 TensorFlow 的 Graph 结构，
    # 而当 Base 框架调用自定义初始化函数 init_from_checkpoint 的时候，
    # TensorFlow 模型的 Graph 结构已经变成 finalized，不再允许修改 Graph 结构。
    # 所以，这个定义必须放在  init_from_checkpoint 函数外面。
    saver = tf.train.Saver(tf.trainable_variables())

    # 返回执行自定义初始化的函数。
    # 该函数必须接收两个参数：
    #   - scafford: tf.train.Scaffold 对象；
    #   - sess: tf.Session 对象。
    def init_from_checkpoint(scaffold, sess):
        print('warm-start from checkpoint {0}'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)
    return init_from_checkpoint

if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn = model_fn,
        gen_init_fn = gen_init_fn)
    distTfRunner.run(train_fn)
