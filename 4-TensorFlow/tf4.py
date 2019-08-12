# placeholders

# Constants     :Constants holds the typical data.
# variables     :Data values will be changed, with respective the functions such as cost_function..
# placeholders  :Training/Testing data will be passed in to the graph.

import tensorflow as tf
x = tf.placeholder(dtype=tf.int32,shape=[3])
y = tf.placeholder(dtype=tf.int32,shape=[3])

sum_x = tf.reduce_sum(x)
prod_y = tf.reduce_prod(y)

other = (sum_x + prod_y)/2

sess = tf.Session()
print("sum_x = ",sess.run(sum_x,feed_dict={x:[4,5,6]}))
print("prod_y = ",sess.run(prod_y,feed_dict={y:[8,8,8]}))
print(sum_x)

# feed_dict can be used also in higher level operations
# where placeholders are not direct inputs for the operation
print("other = ",sess.run(other,feed_dict={x:[1,1,1],y:[0,0,0]}))



sess.close()