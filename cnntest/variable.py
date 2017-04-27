import tensorflow as tf

state = tf.Variable(0,name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state,noe)
update = tf.assign(state,new_value)

init = tf.initialize_all_variable() # must have if define variable


with tf.Session as sess:
	sess.run(init)
	for_in range(3):
		sess.run(update)
		print(sess.run()state)