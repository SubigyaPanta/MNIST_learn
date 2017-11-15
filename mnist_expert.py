import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
sess = tf.InteractiveSession()

# Build model
x   = tf.placeholder(tf.float32, shape=[None, 784]) #shape is optional but helps in debugging
y_  = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# initialize all variables
sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b # our model

#loss / cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# ^^ Note that tf.nn.softmax_cross_entropy_with_logits internally applies
# the softmax on the model's unnormalized model prediction and sums
# across all classes, and tf.reduce_mean takes the average over these sums.

# Train the model
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train.run(feed_dict={x: batch[0], y_: batch[1]})
    sess.run(train, {x: batch[0], y_: batch[1]}) # both are same

# evaluate
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
# after casting true false becomes 1 and 0.
# getting mean will then be adding all 1 and 0 divide by all = % of 1's
# that means total % correct i.e accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})) # both are same