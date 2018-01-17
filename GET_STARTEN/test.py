from __future__ import print_function
import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

print("______________")
sess = tf.Session()
print(sess.run([node1, node2]))

print("______________")
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3):", sess.run(node3))

print("______________")
sess = tf.Session()
print(sess.run([node1, node2]))

print("______________")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

print("_______________")
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

print("_______________")
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

#print(W)
#print(b)
#print(x)
init = tf.global_variables_initializer()
sess.run(init)
print("W node is ", sess.run(W))
print("b node is ", sess.run(b))
#print("x node is ", sess.run(x))
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

print("_______________")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1], y: [0]}))

print("_______________")
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


print("_______________")
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))





