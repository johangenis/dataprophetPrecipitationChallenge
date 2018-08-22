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
### Answer at least the following questions in your discussion:

####  1. What makes this model architecture suitable for the precipitation forecast problem?
The fact that autoencoders is a form of unsupervised learing makes it suitable for the the data presented in the precipitation dataset, which is unlabeled.


####  2. How would your pipeline read image data and feed it into the model?


####  3. How does this model architecture learn from sequence data?


####  4. How does this model architecture generate output sequences?


####  5. How does this model generate images in the outptut layer(s)?


####  6. An overview of at least two different evaluation metrics that could be used to evaluate the model's performance.

Additionally, add the following in your report:

* Give a brief overview of auto-encoders.
  * Autoencoders are a type of neural network that can be used to learn efficient codings of input data. Given some inputs, the network firstly applies a series of transformations that map the input data into a lower dimensional space, which is called the encoder part of the network. 
  * The network then uses the encoded data to try and recreate the inputs, which is called the decoder part of the network. Using the encoder data is compressed into a type that is understood by the network.
* Give a brief overview of variational autoencoders.
  * With variational autoencoders, which also compresses data, it is also possible to generate new images from the input provided to the network. 
* Give a brief overview of generative adversarial networks.
* Optionally, answer the bonus questions below:


To improve your chances, answer the following questions in your report. You may include these answers as part of discussions throughout your report, or answer them directly in a separate section.

1. Discuss two machine learning approaches that would also be able to perform the task. Explain how these methods could be applied instead of your chosen method. 
2. Explain the difference between supervised and unsupervised learning. 
3. Explain the difference between classification and regression. 
4. In supervised classification tasks, we are often faced with datasets in which one of the classes is much more prevalent than any of the other classes. This condition is called class imbalance. Explain the consequences of class imbalance in the context of machine learning. 
5. Explain how any negative consequences of the class imbalance problem (explained in question 4) can be mitigated. 
6. Provide a short overview of the key differences between deep learning and any non-deep learning method.
7. What is the most recent development in AI that you are aware of and what is its application, if any? 
8. Explain how the above development moves our understanding of the field forward.
