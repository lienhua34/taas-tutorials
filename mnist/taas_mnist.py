# coding=utf-8

from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from caicloud.clever.tensorflow import dist_base

tf.app.flags.DEFINE_string("data_dir",
                           "/tmp/mnist-data",
                           "path to mnist dataset.")

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
    if sync:
        num_workers = num_replicas
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_replicas,
            total_num_replicas=num_replicas,
            name="mnist_sync_replicas")
    _train_op = optimizer.minimize(_loss, global_step=_global_step)

    return dist_base.ModelFnHandler(
        global_step = _global_step,
        optimizer = optimizer)

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

if __name__ == '__main__':
    distTfRunner = dist_base.DistTensorflowRunner(
        model_fn = model_fn)
    distTfRunner.run(train_fn)
