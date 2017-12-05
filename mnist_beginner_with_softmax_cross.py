import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
x = tf.placeholder(tf.float32, [None, 784]) # placeholder for x value in data
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b) # holds estimated/predicted/probable digit in image
# y = tf.nn.softmax_cross_entropy_with_logits(_sentinel=tf.matmul(x, W) + b) # same as above but more stable according to docs.
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10]) # placeholder for y value in data

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
count = True
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # if count:
  #   # print('x:', batch_xs, 'y:', batch_ys)
  #   count = False
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # print(type(batch_ys))

# evaluating
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}) )