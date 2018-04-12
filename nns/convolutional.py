import tensorflow as tf
import argparse

from datafeeds.cifar10_data_feed import DataFeed


def main(model_name, restore, log_dir, **kwargs):
    feed = DataFeed()
    
    with tf.Graph().as_default() as graph:
        with tf.name_scope('input'):
            # Input placeholder
            x = tf.placeholder(tf.float32, shape=[None, 3072], name='x')
            # Label placeholder
            y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y')

        # Input Layer
        input_layer = tf.reshape(x, [-1, 32, 32, 3])
        
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
        # Dense Layer - note how we flatten the data by reshaping it
        pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        
        is_training = tf.placeholder(tf.bool)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=is_training)
        
        # Logits Layer
        y_hat = tf.layers.dense(inputs=dropout, units=10)
        
        # Use softmax cross entropy as loss function
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_hat))

        tf.summary.scalar('cross_entropy', cross_entropy)

        # Use Adam optimizer (a variant of gradient descent) to train
        # the model. The training_step will be used to do a packpropagation.
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1.e-4).minimize(cross_entropy)

        # Accuracy finds percentage of correct predictions compared to labels.
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and prepare writer a log directory
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, graph)

        # Prepare saver to store the learned parameters for later use
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if restore:
                saver.restore(sess, model_name)
                
            iterations = 10001
            validate_at = 50 
            batch_size = 100
            checkpoint_at = 1000
            
            for i in range(iterations):
                # Every `validate_at` steps, use validation set to assess model
                # and write summary
                if i % validate_at == 0:
                    xs, ys = feed.validation()
                    # No dropout when validating the model
                    summ, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, is_training: False})
                    writer.add_summary(summ, i)
                    print(f'Accuracy as step {i} is {acc:6.3f}')
                else:
                    xs, ys = feed.next(batch_size)
                    # When we train we want dropout
                    sess.run([train_step], feed_dict={x: xs, y_: ys, is_training: True})

                if i != 0 and i % checkpoint_at == 0:
                    print(f'Saving model at step {i}')
                    saver.save(sess, model_name, global_step=i)

            xs, ys = feed.test()
            acc = sess.run(accuracy, feed_dict={x: xs, y_:ys, is_training: False})
            print(f'Final accuracy on test set: {acc:6.3f}')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convolutional neural network')

    parser.add_argument('--restore', action='store_true',
                        help='If set, variables will be restored from checkpoint')
    parser.add_argument('-m', '--model-name', type=str, default='./checkpoints/convolution.ckpt',
                        help='Name of the checkpoint to restore')
    parser.add_argument('-l', '--log-dir', type=str, default='/tmp/tf/conv',
                        help='Log directory to use for tensorboard')
    args = parser.parse_args()
    main(**vars(args))
