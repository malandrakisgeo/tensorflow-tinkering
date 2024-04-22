import tensorflow as tf;

a = tf.random.uniform([2,2])
b = tf.random.normal([2,2], mean=1,stddev=2)
c = tf.random.normal([2,2])
d = tf.zeros([3,2])

z = tf.random.normal([6,3])
b = tf.ones([3]) # bias vector
y = z + b
aaa = z.numpy()
bbb = y.numpy()

z = y