
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Define loss and optimizer. Add placeholder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#Implement cross-entropy function to determine loss of the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Launch model in an interactive session
sess = tf.InteractiveSession()

#Create an operation to initalize the variables
tf.global_variables_initializer().run()

#Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

#Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Show results
print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))
