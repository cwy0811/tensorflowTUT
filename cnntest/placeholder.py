import tensorfllow as tf

                          # type
input1 = tf.palceholder(tf.float32)
input2 = tf.placeholder(tf.float32)    #must use dictionary to assignment

output = tf.mul(input1,input2)

with tf.Sessio() as sess:
	                                     # Dictionaries
	print(sess.rin(output,feed_dict={input1:[7.],input2:[2.]}))