# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 9:55
# @Author  : Administrator
# @Email   : happiness_ws@163.com
# @File    : knn.py
# @Software: PyCharm
import tensorflow as tf

x = tf.constant([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
xShape = tf.shape(x)

z1 = tf.reduce_mean(x, axis=0)
z2 = tf.reduce_mean(x, axis=1)
z3 = tf.reduce_mean(x, axis=2)

y = tf.constant([[2, 3], [1, 4]])
yshape = tf.shape(y)
ya = tf.expand_dims(y, axis=2)
ye = tf.shape(ya)

with tf.Session() as sess:
    xs, d1, d2, d3, ys, ya, ey = sess.run([xShape, z1, z2, z3, yshape, ya, ye])
    print(xs, " \n d1 =", d1, "  \n d2=", d2, "\n d3 = ", d3, "  \n ys =", ys, "  \n ya = ", ya, "  \n ey = ", ey)

if __name__ == '__main__':
    pass
