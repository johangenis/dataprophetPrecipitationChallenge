# Dataprophet: Precipitation Challenge: 22 August 2018

## The challenge is to write a detailed report on how to train a model to take a sequence of daily precipitation maps as input, and generate precipitation forecast maps for one week (7 days) into the future.

Note that you do not have to implement the model, but that your report should outline the approach you would take. Where applicable, you may provide Python3 TensorFlow code snippets to clarify your approach.
Your chosen model must be a deep learning model, and your report must cover at least the following topics:
###  A detailed overview of the model's architecture, including descriptions of any and all layer types. 
The proposed model is to be a Generative Model using Variational Autoencoding. Variational Autoencoding suits the preciptation forcast problem well, in that the data is unlabeled and new images is expected as predictions from the input data. The network consists of mainly 3 parts - the Encoder, Decoder and Loss Function.
       Defining the Encoder (see code snipped below). The encoder creates objects following a Gaussian Distribution:
   * A vector of means
   * A vector of standard deviations
``` Python
   def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
   return z, mn, sd
  ``` 
#### Defining the decoder:
The decoder will attempt to reconstruct the input images using a series of transpose convolutions:
```Python
   def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])
     return img
  ```
#### Computing losses and enforcing a Gaussian latent distribution:
For computing the image reconstruction loss, the squared difference is used (which could lead to images sometimes looking a bit fuzzy). This loss is combined with the Kullback-Leibler(KL) divergence, which makes sure our latent values will be sampled from a normal distribution.
``` Python
    unreshaped = tf.reshape(dec, [-1, 28*28])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
    loss = tf.reduce_mean(img_loss + latent_loss)
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
 ``` 
### Answer at least the following questions in your discussion:

####  1. What makes this model architecture suitable for the precipitation forecast problem?
The fact that autoencoders is a form of unsupervised learing makes it suitable for the the data presented in the precipitation dataset, which is unlabeled.


####  2. How would your pipeline read image data and feed it into the model?
``` Python
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from local/machine import input_data
    percipitation = input_data.read_data_sets('percipitation_data')

```


####  3. How does this model architecture learn from sequence data?
By designing very expressive observation distributions that can model images, or to condition the simple distribution on a latent variable to produce a hierarchical output distribution. This latter type of model is known as a Variational Auto encoder (VAE).


####  4. How does this model architecture generate output sequences?
Generating new data
``` Python
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
    imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

    for img in imgs:
       plt.figure(figsize=(1,1))
       plt.axis('off')
       plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```


####  5. How does this model generate images in the outptut layer(s)?
New images are generated by sampling values from a unit normal distribution and feeding them to the decoder. 

####  6. An overview of at least two different evaluation metrics that could be used to evaluate the model's performance.
####   Mean Squared Error
MSE takes the average of the square of the difference between the original values and the predicted values. The advantage of MSE being that it is easier to compute the gradient, whereas Mean Absolute Error requires complicated linear programming tools to compute the gradient. As, we take square of the error, the effect of larger errors become more pronounced then smaller error, hence the model can now focus more on the larger errors.
####    Cross Entropy 
Is typically used as the loss metric. The following code calculates cross entropy when the model runs in either TRAIN or EVAL mode:
``` Python
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```

Additionally, add the following in your report:

* Give a brief overview of auto-encoders.
  * Autoencoders are a type of neural network that can be used to learn efficient codings of input data. Given some inputs, the network firstly applies a series of transformations that map the input data into a lower dimensional space, which is called the encoder part of the network. 
  * The network then uses the encoded data to try and recreate the inputs, which is called the decoder part of the network. Using the encoder data is compressed into a type that is understood by the network.
* Give a brief overview of variational autoencoders.
  * With variational autoencoders, which also compresses data, it is also possible to generate new images from the input provided to the network. 
* Give a brief overview of generative adversarial networks.
* Optionally, answer the bonus questions below:


To improve your chances, answer the following questions in your report. You may include these answers as part of discussions throughout your report, or answer them directly in a separate section.

## Due to time constraints after working the whole weekend on Production issues with the Standard Banks databases, the below was not attempted.

1. Discuss two machine learning approaches that would also be able to perform the task. Explain how these methods could be applied instead of your chosen method. 
2. Explain the difference between supervised and unsupervised learning. 
3. Explain the difference between classification and regression. 
4. In supervised classification tasks, we are often faced with datasets in which one of the classes is much more prevalent than any of the other classes. This condition is called class imbalance. Explain the consequences of class imbalance in the context of machine learning. 
5. Explain how any negative consequences of the class imbalance problem (explained in question 4) can be mitigated. 
6. Provide a short overview of the key differences between deep learning and any non-deep learning method.
7. What is the most recent development in AI that you are aware of and what is its application, if any? 
8. Explain how the above development moves our understanding of the field forward.
