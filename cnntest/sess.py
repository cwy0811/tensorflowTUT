import tensorflow as tf

matrix1 = tf.constant([3,3])  # 1x2
matrix2 = tf.constant([[2],   
                        [2]])  # 2x1
product = tf.matmul(matrix1,matrix2)

# method 1
sess = tf.Session()
result = ses.run(product)
print(result)
sess.close()   # need to close sess

# method 2
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)   # need not to close sess
                   