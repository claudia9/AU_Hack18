import tensorflow as tf
import argparse

from data_feed import DataFeed

def main(model_name, restore, log_dir, **kwargs):
    feed = DataFeed()
    
    with tf.Graph().as_default() as graph:
        with tf.name_scope('input'):
            # Input placeholder
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            # Label placeholder
            y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y')

        with tf.name_scope('FC1'):
            # Create variables
            W1 = tf.Variable(tf.truncated_normal((784,500), stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=(500,)))

            # Add variable summaries
            tf.summary.histogram('w1-histogram', W1)
            tf.summary.histogram('b1-histogram', b1)
            
            # Linear layer
            pre1 = tf.matmul(x, W1) + b1
            
            # Add non-linearity
            hidden1 = tf.nn.relu(pre1)
            
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            dropped = tf.nn.dropout(hidden1, keep_prob)

        with tf.name_scope('FC2'):
            # Create variables
            W2 = tf.Variable(tf.truncated_normal((500,10), stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=(10,)))

            # Add variable summaries
            tf.summary.histogram('w2-histogram', W2)
            tf.summary.histogram('b2-histogram', b2)
            
            # Linear layer
            y_hat = tf.matmul(dropped, W2) + b2
            
        with tf.name_scope('cross_entropy'):
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_hat))
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            # Training
            train_step = tf.train.AdamOptimizer(1.e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, graph)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            iterations = 10000
            save_at = 50
            batch_size = 100

            for i in range(iterations):
                # Every `save_at` steps, use validation set to assess model
                # and write summary
                if i % save_at == 0:
                    xs, ys = feed.validation()
                    summ, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, keep_prob: 1.0})
                    writer.add_summary(summ, i)
                    print(f'Accuracy as step {i} is {acc}')
                else:
                    xs, ys = feed.next(batch_size)
                    sess.run([train_step], feed_dict={x: xs, y_: ys, keep_prob: 0.5})
                    
            xs, ys = feed.test()
            acc = sess.run([accuracy], feed_dict={x: xs, y_:ys, keep_prob: 1.0})
            print(f'Final accuracy on test set: {acc}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Linear neural network')

    parser.add_argument('--restore', action='store_true',
                        help='If set, variables will be restored from checkpoint')
    parser.add_argument('-m', '--model-name', type=str, default='./fully_connected.ckpt',
                        help='Name of the checkpoint to restore')
    parser.add_argument('-l', '--log-dir', type=str, default='/tmp/tf/fc',
                        help='Log directory to use for tensorboard')
    args = parser.parse_args()
    main(**vars(args))
