# constants and variables
import tensorflow as tf
x = tf.constant(50)
y = tf.Variable(100)
z = tf.Variable(x+y)

with tf.Session() as sess1:

    # executing only this line wont work
    # because we have to initialize variables before using them inside the session
    # print(sess1.run(z))

    # initializing
    sess1.run(tf.global_variables_initializer())
    print(sess1.run(z))

