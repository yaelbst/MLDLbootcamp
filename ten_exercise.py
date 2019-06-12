import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Hi")

node1 = tf.constant(5, name="node1")
node2 = tf.constant(11, name="node2")

node3 = node1 + node2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

a = x + y

sess = tf.Session()
sess.run(tf.initialize_all_variables())
result = sess.run(node3)
result2 = sess.run(a, {x:[12],y:[8.5]})

print(result)

print(result2)

sess.close()

x = tf.placeholder(tf.float32, shape=[4])
W = tf.Variable([7,7,7,7], dtype=tf.float32)
b = tf.Variable([8,8,8,8], dtype=tf.float32)

s = W + b

linear_model = (W * x)+ b

y = tf.placeholder(tf.float32, shape=[4])

L = tf.square(linear_model - y)
loss = tf.reduce_sum(L)

sess2 = tf.Session()
sess2.run(tf.initialize_all_variables())
res = sess2.run(s)
print(res)
res2 = sess2.run(linear_model, {x:[1,2,3,4]})
print(res2)
res3 = sess2.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]} )
print(res3)
sess2.close()

assign_op1 = W.assign([-1,-1,-1,-1])
assign_op2 = b.assign([1,1,1,1])

sess3 = tf.Session()
sess3.run(tf.initialize_all_variables())
sess3.run(assign_op1)
sess3.run(assign_op2)
res4 = sess3.run(loss,{x:[1,2,3,4], y:[0,-1,-2,-3]})
print(res4)

sess3.close()

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

assign_op3 = W.assign([7,7,7,7])
assign_op4 = b.assign([8,8,8,8])

with tf.Session() as sess5:
    sess5.run(tf.initialize_all_variables())
    sess5.run(assign_op3)
    sess5.run(assign_op4)
    values = {x: [1,2,3,4],
              y: [0,-1,-2,-3]}
    loss_vec = []
    # plt.ylabel("Loss Magnitude")
    # plt.xlabel("Training iterations")
    for t in range(1000):
        loss_val,_ = sess5.run([loss, train],feed_dict=values)
        loss_vec.append(loss_val)
        # if t%10 == 0:
        #     print(loss_val)
    print(W.eval())
    print(b.eval())
    print(loss_val)
    plt.plot(loss_vec)
    plt.show()

sess5.close()