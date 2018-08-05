
# coding: utf-8

# In[137]:

import pandas as pd
import os
import numpy as np


# In[138]:

def opencsv():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    train_data = train.values[0:, 1:]
    train_lable = train.values[:, 0]
    test_data = test.values[0:, 0:]
    return train_data, train_lable, test_data


# In[139]:

import numpy as np


# In[140]:

np.random.seed(13)

print(os.getcwd())

x_vals, train_lable, test_data = opencsv()


# In[141]:

print(len(train_lable))
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = train_lable[train_indices]
y_vals_test = train_lable[test_indices]


# ### 计算距离L1

# In[142]:

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()


# In[143]:

x_input = tf.placeholder(shape=[None, 28 * 28], name="x_input", dtype=tf.float32)
y_input = tf.placeholder(shape=[None], name="y_input", dtype=tf.float32)
test_input = tf.placeholder(shape=[None,28 * 28], name="test_input", dtype=tf.float32)
top_k_indices_input=tf.placeholder(shape=[None,5],name="top_k_indices",dtype=tf.int32)


# In[144]:

print (len(test_data))


# In[145]:

d_temp = tf.add(x_input, tf.negative(test_input))
distance = tf.reduce_sum(tf.abs(d_temp),reduction_indices=1)
#tf.expand_dims(test_input, axis=1)


# In[146]:

top_k_xvals,top_k_indices = tf.nn.top_k(tf.negative(distance), k=5)
# shape_top_k_xvals = tf.shape(top_k_xvals)
# shape_top_k_indices = tf.shape(top_k_indices)

# x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
# x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))  # shape = [k,k]
# x_val_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)  #
# 
# count = tf.unique_with_counts(tf.cast(top_k_yvals, dtype=tf.int32))

# pre = tf.argmax(top_k_xvals, axis=1)


# In[167]:

top_k_yvals = tf.gather(y_input, top_k_indices)

count = tf.bincount(tf.cast(top_k_yvals,dtype=tf.int32))

pre_y = tf.argmax(count)


# In[169]:

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess.run(init)
batch_size = 1
target_size = 5000
num_loops = int(np.ceil(len(x_vals_test) / batch_size))
num_loops = 1000
# pre_dic = []
# target_dict=[]
acc = []
for i in range(num_loops):
    min_index = i * batch_size
    max_index = min((i + 1) * batch_size, len(x_vals_test))
    random_index = np.random.choice(len(x_vals_train),target_size)
    pre = sess.run(pre_y, feed_dict={x_input:x_vals_train[random_index], test_input: x_vals_test[min_index:max_index, :],
                                 y_input: y_vals_train[random_index]})
#     print(np.argmax(pre),"  ",y_vals_test[min_index])
#     pre_y = np.argmax(pre)
#     print(pre)
#     pre_dic.append(pre)
#     target_dict.append(y_vals_test[min_index])
    if pre == y_vals_test[min_index]:
        acc.append(1)
    else:
        acc.append(0)

print(np.mean(acc))

#     pre_dic.append(pre)

# print(pre_dic)
#     print(y_vals_train[random_index])
#     print(k_index)
#     for j in range(len(k_index)):
#         print(k_index[j])
    
#     print("shape ",y_vals_test[k_index])
#     print(y_vals_test[min_index:max_index], " pre ", np.)


# In[ ]:



