import tensorflow as tf
import argparse

from datafeeds.mnist_data_feed import DataFeed

def main(model_name, restore, logdir, **kwargs):
    feed = DataFeed()
    
    with tf.Graph().as_default() as graph:
        with tf.name_scope('input'):
            # Input placeholder
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            # Label placeholder
            y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y')

        dense = tf.layers.dense(inputs=x,units=500, activation=tf.nn.relu)

        is_training = tf.placeholder(tf.bool)
        dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=is_training)

        y_hat = tf.layers.dense(inputs=dropout, units=10)

        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)

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

        # Merge all the summaries and write them out to the logdir
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, graph)

        # Prepare saver to store the learned parameters for later use
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if restore:
                saver.restore(sess, model_name)
                
            iterations = 10001
            validate_at = 50
            batch_size = 100
            checkpoints = 10
            checkpoint_at = iterations // checkpoints
            
            for i in range(iterations):
                # Every `validate_at` steps, use validation set to assess model
                # and write summar
                xs, ys = feed.next(batch_size)
                # When we train we want dropout
                sess.run([train_step], feed_dict={x: xs, y_: ys, is_training: True})
                if i % validate_at == 0:
                    xs, ys = feed.validation()
                    # No dropout when validating the model
                    summ, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, is_training: False})
                    writer.add_summary(summ, i)
                    print(f'Accuracy as step {i} is {acc:6.3f}%')
                
                if i != 0 and i % checkpoint_at == 0:
                    print(f'Saving model at step {i}%')
                    saver.save(sess, model_name, global_step=i)

            xs, ys = feed.test()
            acc = sess.run(accuracy, feed_dict={x: xs, y_:ys, is_training: False})
            print(f'Final accuracy on test set: {acc:6.3f}%')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Linear neural network')

    parser.add_argument('--restore', action='store_true',
                        help='If set, variables will be restored from checkpoint')
    parser.add_argument('-m', '--model-name', type=str, default='./checkpoints/dense.ckpt',
                        help='Name of the checkpoint to restore')
    parser.add_argument('-l', '--logdir', type=str, default='/tmp/tf/dense',
                        help='Log directory to use for tensorboard')
    args = parser.parse_args()
    main(**vars(args))
