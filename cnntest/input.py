import tensorflow as tf

# read the imagine data 
reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer(['cat.jpg']))
image0 = tf.image.decode_jpeg(value)





