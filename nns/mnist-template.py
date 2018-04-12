import tensorflow as tf
import argparse

from datafeeds.mnist_data_feed import DataFeed


def main(model_name, restore, log_dir, **kwargs):
    print('Starting training of mnist network')
    print(f'Writing summaries to {log_dir}')
    
    feed = DataFeed()
    
    with tf.Graph().as_default() as graph:
        with tf.name_scope('input'):
            # Input placeholder
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            # Label placeholder
            y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y')

        # Input Layer
        input_layer = tf.reshape(x, [-1, 28, 28, 1])

        #############################################
        #           BUILD YOUR MODEL HERE           #
        #############################################
        

        # Logits Layer
        y_hat = # TODO: build model
        
        #############################################
        
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

        # Histograms of learned parameters
        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)

        # Merge all the summaries and prepare writer a log directory
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, graph)

        # Prepare saver to store the learned parameters for later use
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if restore:
                print(f'Restoring model from {model_name}')
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
    parser = argparse.ArgumentParser(description='MNIST neural network')

    parser.add_argument('--restore', action='store_true',
                        help='If set, variables will be restored from checkpoint')
    parser.add_argument('-m', '--model-name', type=str, default='./checkpoints/mnist-template.ckpt',
                        help='Name of the checkpoint to restore')
    parser.add_argument('-l', '--log-dir', type=str, default='/tmp/tf/mnist',
                        help='Log directory to use for tensorboard')
    args = parser.parse_args()
    main(**vars(args))
