import tensorflow as tf
import argparse

from datafeeds.mnist_data_feed import DataFeed

def main(model_name, restore, **kwargs):
    feed = DataFeed()
    
    with tf.Graph().as_default() as graph:
        # Input placeholder
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        # Label placeholder
        y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y')

        dense = tf.layers.dense(inputs=x,units=500, activation=tf.nn.relu)

        is_training = tf.placeholder(tf.bool)
        dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=is_training)

        y_hat = tf.layers.dense(inputs=dropout, units=10)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_hat))

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1.e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            if restore:
                saver.restore(sess, model_name)
                
            for i in range(10001):
                xs, ys = feed.next(100)
                sess.run([train_step], feed_dict={x: xs, y_: ys, is_training: True})

                if i % 100 == 0:
                    xs, ys = feed.validation()
                    acc = sess.run(accuracy, feed_dict={x: xs, y_: ys, is_training: False})
                    print(f'Accuracy as step {i} is {acc:6.3f}%')

                if i != 0 and i % 1000 == 0:
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
    parser.add_argument('-m', '--model-name', type=str, default='./checkpoints/dense-simple.ckpt',
                        help='Name of the checkpoint to restore')
    args = parser.parse_args()
    main(**vars(args))
