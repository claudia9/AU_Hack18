# AUHack Tensorflow workshop
This project is created for the **AUHack 2018** event. It indends to
  give a comprehensible introduction to *Tensorflow* and
  *TensorBoard*. A basic knowledge about Python and Neural networks is
  assumed.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project structure](#project-structure)
- [About Tensorflow](#about-tensorflow)
- [About TensorBoard](#about-tensorboard)
- [Models](#models)
- [Slides from AUHack](#slides-from-auhack)

##  Requirements
 - Python 3.6
   - Tensorflow
   - Numpy

## Installation
If you don't currently have Python version 3.6 installed, an easy way
to obtain python is by
installing
[Anaconda](https://conda.io/docs/user-guide/install/index.html)
and then install an environment using the following command:

```bash
> cd /path/to/this/repo
> conda create --name [envname] python=3.6 -y --file requirements.txt
```

This will install a python environment with the needed packages for
this project. The packages are listed in the `requirements.txt`
file. `[envname]` is the name of the enviromnemt, e.g., `auhack`.

Before executing any python scripts or launching tensorboard, you will
then have to activate the new environment:

**Linux, MacOS**:
```bash
> source activate [envname]
```

**Windows**:
```bash
> activate [envname]
```

Now you are good to go.

## Project structure
The project is structured as follows. In this readme we give an
introduction to Tensorflow and Tensorboard. In the folder `nns` we
have created two neural networks. Further down this page is
instructions on how to train the networks.

## About Tensorflow ##
In this section we briefly give a simple overview of the concepts of
tensorflow. If you are already familiar with tensorflow, you can
safely skip this part.

[Tensorflow](https://www.tensorflow.org/) is a python library, which
is good at exactly what the name indicates; orchestrating flows of
tensors (multi-dimensional vector). In order to orchestrate tensors,
one needs to understand three concepts, namely, *tensors*, *graphs*,
and *sessions*. We will cover them conceptually here and in greater
detail in the section about the first (and simpler model) included in
this project.

### Graphs ###

In Tensorflow, you build graphs containing all the information about
how different tensors "interact" with each other. The graph it self is
not doing any computations. It is only there to structure in what
order computations have to be carried out.

A simple example could be the simple expression `a(b + c)`. This
expression can be represented as the following graph:

![readme-graphics/simple-graph.png](readme-graphics/simple-graph.png)

Note, how each vertex in the graph is an operation (`add` og `mul` in this
case) how data (tensors) flow on the edges between the vertices.

### Tensors ###

The data flowing along the edges is denoted tensors. When building
graphs, we almost always apply operations to tensors in order to get a
new tensor. Tensors are volumes of arbitrary dimensions. An image
would, for example, be a 3D-tensor, since it has a height, a width,
and depth (color channels). 

As an example of how we work with tensors, let's build the graph from
above using tensorflow:

```python
import tensorflow as tf

a = tf.placeholder(dtype=tf.float32, shape=(1,), name='a')
b = tf.placeholder(dtype=tf.float32, shape=(1,), name='b')
c = tf.placeholder(dtype=tf.float32, shape=(1,), name='c')

bc = b + c
out = tf.multiply(a, bc)
```

In the code above, a couple of things are happening. Let's walk
through them one at a time.
1. We start by creating three placeholders `a`, `b`, and `c`. The reason why
   we do this is that we don't wish to specify what the exact values
   are quite yet. We just need to tell tensorflow that we will later provide the
   values of a, b, and c, each being a single float. The shape is the
   shape of the tensor. `(1,)` means a vector of length 1. Had we
   written, e.g., `(2, 3)` it would have meant a matrix with two rows
   and three columns. We could also have used a single scalar by using
   the shape `()`.
2. Next we apply an operation to the two tensors `b` and `c` in order to
   get a new tensor `bc`. Note how the + is translated into a tensorflow
   operation. We could also have used `bc = tf.add(b, c)`.
3. Finally we multiply the tensor `a` with the tensor `bc` to get the
   final output.

Apart from placeholders, there are two other ways of holding data. We
can use constants and variables. Variables are of
particular interest since they can be altered during a session. We use
variables to learn the parameters of our deep neural network.

Now, we have built a graph, without calculating anything yet. In fact,
none of the tensors knows anything about any actual values. The
calculation with actual values is the purpose of the session.

### Sessions ###

In order to execute a graph, two important things have to happen.
1. We need the data to do the calculations on, i.e., we need the
   data to fill into the placeholders.
2. We need to know what we want calculated, i.e., we need to know the
   tensor that we wish to be evaluated.

In the case of the running example, we may wish to know what a(b + c)
is when  a=1, b=2, and c=3. Then, we need to evaluate the tensor
`out`.

The session is what we use to evaluate `out` given some values for the
placeholders. We do this in the following way:

```python
with tf.Session() as sess:
	result = sess.run(out, feed_dict={a: [1.0], b: [2.0], c: [3.0]})
```

Here, there are three observations to do:
1. The `sess.run(...)` is what evaluates the tensor `out`. It returns
   the actual result with actual values, when executed.
2. We construct a dictionary with placeholders as keys, and actual
   values as values and feed it into the tensorflow session through
   the keyword `feed_dict`. All placeholders used for the evaluation
   of `out` must ofcourse be specified.

This should provide the proper intuition of how graphs and sessions
differ. The reason why we need both sessions and graphs is
performance. When we run a session, the graph and the data is
transfered to a very optimized C library in order to obtain higher
performance than what python it self can provide. If there are
variables in you graph, they have to be
initialized as the first thing before evaluating tensors with the
session:

```python
...
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	...
```


A final note; it may be that we also want the intermediate result `b +
c`, in which case we could have run the following session instead:


```python
with tf.Session() as sess:
	result, intermediate = sess.run([out, bc], feed_dict={a: [1.0], b: [2.0], c: [3.0]})
```

### Usual structure of code

```python
import tensorflow as tf

with tf.Graph().as_default() as graph:
	... do graph building stuff ...

	with tf.Session() as sess:
		... run graph stuff ...
```

### Layers ###
With the `tensorflow.layers` library it is very easy to build a grap
with standard neural network layers. In this subsection we will give a
couple of examples used in the workshop.

#### Dense layer ####

Dense layers are also known as fully connected layers. In tensorflow
we can make a linear layer as follows.

```python
with tf.Graph().as_default() as graph:
	x = tf.placeholder(tf.float32, (None, 784), name='x')
	dense = tf.layers.dense(
		inputs=x,
		units=500,
		activation=tf.nn.relu)
```

This code will make a dense layer, that takes a matrix of samples of
size vectors of size 784 as inputs and outputs vectors of
size 500. The layer will multiply a 784 by 500 matrix to the input,
add a bias vector, and apply the ReLU nonlinearity.

#### Convolutional layer ####
A convolutional layer is very commonly used in image recognition and
is also very easily applied in tenforflow:

```python
with tf.Graph().as_default() as graph:
	x = tf.placeholder(tf.float32, (None, 784), name='x')
	input_layer = tf.reshape(x, (-1, 28, 28, 1)

	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)
```

In the above code we first apply a reshape to the input in order to
transform each sample from a vector of shape `(784,)` to a 3D-volume of
shape `(28, 28, 1)` (think gray scale image, i.e., height, width,
channels). Then we apply the 2D convolution with 32 5 by 5 filters,
using 'same' padding, which mean padding the samples with zeros in
order to obtain the same height and width of the output. Finally, a
relu is applied. 

The output of the operation will be of shape `(28, 28, 32)`.

#### Max-pooling ####
Max-pooling is a summarizing layer typically used right after
convolutional layers. We can apply max-pooling as follows:

```python
pool = tf.layers.max_pooling2d(
	inputs=conv1,
	pool_size=[2, 2],
	strides=2)
```

This will add a max-pooling layer with a 2 by 2 kernel side and a
stride of 2. When the stride is two, the height and widht dimension
will be halved in the output, e.g., if a max-pooling is applied to
samples of shape `(28, 28, 3)`, the output shape will be `(14, 14, 3)`.

#### Dropout ####

Dropout is a commonly used regularization, which helps the neural
network avoid overfitting:

```python
is_training = tf.placeholder(tf.bool)
dropout = tf.layers.dropout(
	inputs=input_layer,
	rate=0.4,
	training=is_training)
```

Here we first make a placeholder, that we can use, when running the
session, to tell the layer if we are training or not. When we are
training, we want dropout but when we validate or test the model, we
do not want dropout. The `rate` tells how many connections between the
preceding and the following layer to drop.

The input and output shape will be the same.

### Saving and restoring models ###
When a model have been trained, we want to save it in order to beable
to reuse it at a later point. For this, we can use a saver object:

```python
...
# Create a saver.
saver = tf.train.Saver(tf.trainable_variables())
# Launch the graph and train, saving the model every 1,000 steps.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(1000000):
		sess.run(..training_op..)
		if step % 1000 == 0:
			# Append the step number to the checkpoint name:
			saver.save(sess, 'path/to/model', global_step=step)
```
The `global_step=step` argument will save a new model at
`path/to/model-[step]`. This is useful if the model ends up
overfitting. Then we can restore older models that didn't overfit
yet.

This will store the learned parameters and enable us to restore the
variables in an other sessions:

```python
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(s, model_file)
	# Now the variables are initialized to the stored values
```

## About TensorBoard ##
This section gives a shot introduction to tensorboard. If you are
already familiar with tensorboard, then you can skip this section.

When we build and train networks, it can be very valuable in terms of
debugging and evaluation to be able to visualize the network graph as
well as different variables during training. For this, we use
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).

We can use tensorboard to many things, but in this workshop, we will
primarily use it to visualize graphs, accuracies, and parameter
distributions. 

To use tensorboard, we have to initialize a file summary file writer
with a destination to where is can write the summary:

```python
writer = tf.summary.FileWriter('/tmp/summaries/1')
```

and then we can add different summaries.
 - **Graphs**: We would like to visualize the graph as in the example
   above.
 - **Scalars**: We may wish to follow, e.g., the accuracy or cross entropy of our
   network while training.
 - **Histograms**: We can follow how the distribution of our learned
   parameters evolve over time.

We can add such summaries as follows:

```python
with tf.graph().as_default() as graph:
	# Build summaries into your graph
	...
	tf.summary.scalar('mean', var)
	...
	tf.summary.histogram('histogram', var)

	# Construct a merged summary to be run in a session
	merged = tf.summary.merge_all()

	writer.add_graph(graph)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		... 
		# Execute summary
		summary = sess.run(merged, feed_dict=...)
		writer.add_summary(summary, i)
```

Afterwards, summaries can be visualized by running following command:

```bash
> tensorboard --logdir /tmp/summaries/1
```
Now we can use the visualizations to assess one or more models.

The tensorboard visualizations could look something like the following:

![readme-graphics/accuracy.png](readme-graphics/accuracy.png)
![readme-graphics/cross_entropy.png](readme-graphics/cross_entropy.png)
![readme-graphics/fc-graph.png](readme-graphics/fc-graph.png)

## Models ##
The models created for this workshop can be found in the folder
`nns`. There are two models. A simpler model used to recognize
[MNIST](http://yann.lecun.com/exdb/mnist/) images and a deeper
convolutional network trained on
the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data set.

### MNIST model ###

The MNIST model is found in `nns/dense.py`. It consists of two dense
layers, a dropout layer and a softmax layer.

To run the model, navigate to the folder and run the script:

```bash
> cd nns
> python dense.py
```

This will download the MNIST data and start training the model.

For more options see
```bash
> python dense.py --help
```

### CIFAR-10 ###

The CIFAR-10 model is found in `nns/convolutional.py`. It consistes of
two convolutional layers, two max-pooling layers and two dense
layers.

To run the model, navigate to the folder and run the script:

```bash
> cd nns
> python convolutional.py
```

This will download the CIFAR-10 data and can be restored to an
accuracy of about 70% by running the following command:

```bash
> python convolutional.py --restore
```

### Data feeds ###
In order to make the process of building neural networks easier at
this workshop, two data feeds have been made. They are located in the
`nns/datafeeds` folder.

The purpose of the datafeeds is to provide data for training and
validation. They download the MNIST and CIFAR-10 data if it is missing
and the supports three operations. 

1. `feed.next(batch_size)` returns a batch of training images and
   their corresponding labels. When all samples have been returned,
   the training set is reshuffled at next call to next.
2. `feed.validation()` and `feed.test()` returns the validation and
   test set, respectively, in a similar manner as the `next`
   function.

### Templates ###
A template with all the boilerplate code can be found in
`nns/mnist-template.py` and `nns/cifar-template.py`.


## Slides from AUHack ##
The slides for this workshop can be
found
[here](https://docs.google.com/presentation/d/1BWe6HsePKh6KCYNn09VmmX6zHPskM7S2l8YuCbiF_3Q/edit?usp=sharing).

## License ##

*MIT License*

Copyright (c) 2018 Frederik Hvilshøj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
