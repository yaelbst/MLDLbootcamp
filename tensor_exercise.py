import numpy as np
import tensorflow as tf

node1 = tf.constant(5, name="node1")
node2 = tf.constant(11, name="node2")

node3 = node1 + node2

sess = tf.Session()
sess.run(tf.initialize_all_variables())
result = sess.run(node3)

print(result)

