# interactive sessions
# difference is we do not have to pass the session object around

import tensorflow as tf

a = tf.constant(4)
b = tf.constant(20)
c = a + b

intSess = tf.InteractiveSession()
print(c.eval())

intSess.close()
