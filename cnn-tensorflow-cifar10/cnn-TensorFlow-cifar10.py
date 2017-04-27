# coding: utf-8

# Python2
# from __future__ import print_function, division
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import cPickle as pickle
# import seaborn

# Python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import seaborn


def unpickle(filename):
    '''解压数据'''
    # Python2
    # with open(filename) as f:
    #     d = pickle.load(f)
    #     return d
    
    # Python3
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d


def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


# 训练数据集
data1 = unpickle('cifar10-dataset/data_batch_1')
data2 = unpickle('cifar10-dataset/data_batch_2')
data3 = unpickle('cifar10-dataset/data_batch_3')
data4 = unpickle('cifar10-dataset/data_batch_4')
data5 = unpickle('cifar10-dataset/data_batch_5')
X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
y_train = onehot(y_train)
# 测试数据集
test = unpickle('cifar10-dataset/test_batch')
X_test = test['data'][:5000, :]
y_test = onehot(test['labels'])[:5000, :]
del test

print('Training dataset shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Testing dataset shape:', X_test.shape)
print('Testing labels shape:', y_test.shape)

with tf.device('/cpu:0'):
    
    # 模型参数
    learning_rate = 1e-3
    training_iters = 500
    batch_size = 50
    display_step = 5
    n_features = 3072  # 32*32*3
    n_classes = 10
    n_fc1 = 384
    n_fc2 = 192

    # 构建模型
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])


    w = {
        'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=5e-2)),
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1)),
        'fc1': tf.Variable(tf.truncated_normal([8*8*64, n_fc1], stddev=0.04)),
        'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
        'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=1/192.0))
    }
    b = {
        'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64])),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
        'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
        'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
        'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
    }

    x4d = tf.reshape(x, [-1, 32, 32, 3])
    # 卷积层 1
    conv1 = tf.nn.conv2d(x4d, w['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b['conv1'])
    conv1 = tf.nn.relu(conv1)
    # 池化层 1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # LRN层，Local Response Normalization
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    # 卷积层 2
    conv2 = tf.nn.conv2d(norm1, w['conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b['conv2'])
    conv2 = tf.nn.relu(conv2)
    # LRN层，Local Response Normalization
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    # 池化层 2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    reshape = tf.reshape(pool2, [-1, 8*8*64])
    dim = reshape.get_shape()[1].value
    fc1 = tf.add(tf.matmul(reshape, w['fc1']), b['fc1'])
    fc1 = tf.nn.relu(fc1)
    # 全连接层 2
    fc2 = tf.add(tf.matmul(fc1, w['fc2']), b['fc2'])
    fc2 = tf.nn.relu(fc2)
    # 全连接层 3, 即分类层
    fc3 = tf.add(tf.matmul(fc2, w['fc3']), b['fc3'])

    # 定义损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 评估模型
    correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    c = []
    total_batch = int(X_train.shape[0] / batch_size)
    for i in range(training_iters):
        avg_cost = 0
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            _, co = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += co
    
        c.append(avg_cost)
        if (i+1) % display_step == 0:
            print("Iter " + str(i+1) + ", Training Loss= " + "{:.6f}".format(avg_cost))
    
    print("Optimization Finished!")

    # Test
    test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print("Testing Accuracy:", test_acc)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('lr=%f, ti=%d, bs=%d, acc=%f' % (learning_rate, training_iters, batch_size, test_acc))
    plt.tight_layout()
    plt.savefig('cnn-tf-cifar10-%s.png' % test_acc, dpi=200)
