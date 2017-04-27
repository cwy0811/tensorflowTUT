import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(tf.floot32)
y_data = x_data*0.1 + 0.3

###  creat tensorflow structure start  ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 = learning rate 0~1
train = optimizer.minimize(loss)

init = tf.initializer_all_variable()  #  Initialization
###  creat tensorflow structure  end ###

sess = tf.Session()
sess.run(init)     #very important 

for step in range(201):
	sess.run(train)
	if step % 20 ==0:
		print(step,sess.run(Weights),sess.run(biases))