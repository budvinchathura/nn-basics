import tensorflow as tf
a = tf.constant(6,name="constant_a")        #name is optional
b = tf.constant(value=20)
# prints values as 0
# because tensors are not evaluated yet
print(a)
print(b)


c=tf.multiply(a,b)

# we need to start a tf session to evaluate above commands
sess = tf.Session()

# these two are not essesntial because "a" and "b" are evaluated when evaluating "c"
# sess.run(a)
# sess.run(b)
# print(sess.run(c))

#only this line is sufficient
print(sess.run(c))

sess.close()