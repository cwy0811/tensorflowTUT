#!/usr/bin/python
#coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
12-13课时内容，添加及创建神经网络。
'''

#添加一个神经层
def add_layer(inputs, in_size, out_size, activation_function = None):
#添加层 inputs为传入数据， in_size传入行数， out_size传入列数， 激励函数默认为None，即为一个线性函数
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	#定义的Weighs为一个随机变量，生成初始值比全部为0好，形状为in_size行 & out_size列 的矩阵
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	#定义biases初始值为0.1的1行，out_size列的向量
	Wx_plus_b  = tf.matmul(inputs, Weights) + biases
	#Weight*x+biases
	if activation_function is None:
	#如果是线性的关系，那么不必再加activation_function，保持现状及就好啦
		outputs = Wx_plus_b
	else:
	#如果不是线性的关系
		outputs = activation_function(Wx_plus_b)
	return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#-1到1这个区间，300个单位即300行，300个例子？
noise = np.random.normal(0, 0.05, x_data.shape)
#noise方差为0.05， mean为0？ 格式为x_data的格式
y_data = np.square(x_data) - 0.5 + noise
#y_data = x_data*x_data -0.5 + noise 其中noise为噪点

xs = tf.placeholder(tf.float32,[None, 1])
#None表示给出多少个simple都可以，x_data 维数为1？
ys = tf.placeholder(tf.float32,[None, 1])

#隐藏层
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
#输入x_data，in_size即x_data的size即1，10个神经元，激励函数为relu

#输出层
prediction = add_layer(l1, 10, 1, activation_function = None)
#输出层的inputs即为隐藏层的输出outputs=l1, in_size为隐藏层的size=10，out_size为y_data的size=1

#预测输出值与真实值的差别，对每一个例子误差求和，并且求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

#见之前的练习说明
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#以梯度下降，学习率为0.1的速度，最小化误差loss


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#生成一个图片框
fig = plt.figure()
#连续性的画图，编号为(1,1,1)
ax = fig.add_subplot(1,1,1)
#一点的形式画图
ax.scatter(x_data, y_data)
#连续的输入
plt.ion()
#输出
plt.show()


for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i%50==0:
		print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		try:
		#抹除lines的第一个线段
			ax.lines.remove(lines[0])
		#如果没有pass
		except Exception:
			pass
		#显示预测的数据,prediction与xs有关，所以要feed_dict
		prediction_value = sess.run(prediction, feed_dict={xs:x_data})
		#将prediction的值画上去，用曲线的形式
		#x轴数据为x_data，y轴为prediction，为红色的线，宽度为3
		lines = ax.plot(x_data, prediction_value, 'r-', lw=3)
		#画的过程中暂停0.1s
		plt.pause(1)
		
























































