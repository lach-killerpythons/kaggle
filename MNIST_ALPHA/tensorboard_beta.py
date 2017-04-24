"""
This creates a tensorboard of a gradient descent implementation for the MNIST dataset.
Implementation is done in tf version 1.0
Hosting tensorboard on local machine allows interaction with the tensorboard.
Interaction includes: diagrams, histograms, cost functions, ect..
"""



import input_data # must have this file in directory
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf #Version 1.0+


graphdump = '/Users/Alien/Documents/python/data ventures/MNIST_ALPHA/graphs' #change this for your local machine


learning_rate= 0.01
training_iteration = 30
batch_size = 100
display_step = 2

#TF graph input
x = tf.placeholder("float", [None, 784]) #mnist data image = 784 vector
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition = 10 classes

#degine weights (probabilities)
W = tf.Variable(tf.zeros([784, 10])) #wieghts
b= tf.Variable(tf.zeros([10])) #bias

with tf.name_scope("Wx_b") as scope: #name scope
    model = tf.nn.softmax(tf.matmul(x,W) + b) #matrix multiplacation

# add summary ops to collect data
#w_h = tf.histogram_summary("weights", W)
w_h = tf.summary.histogram("weights", W)
#b_h = tf.histogram_summary("biases", b)
b_h = tf.summary.histogram("biases", b)
with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar("cost_function", cost_function)


with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

merged_summary_op = tf.summary.merge_all() #tf.merge_all_summaries() ~ old

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(graphdump, graph_def=sess.graph_def)
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost+=sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})
            summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch*i)
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration+1), "cost=", "{:.9f}".format(avg_cost))

    print("complete!")
    predictions = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ("accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


#to run board [1] In Cmd (change for local graph dir) [2] In Browser (change for local machine name/IP address)
#[1]# tensorboard --logdir=/Users/Alien/Documents/python/data ventures/MNIST_ALPHA/graphs --port 6006
#[2]# userhost:6006
