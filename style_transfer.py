import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc

import pandas as pd

#intializing output directory
output_dir="./output" #intializing output directory
image_for_style="./"+fest+".jpg"#path of style image
content_image=path#path of content image

#intializing parameters of output image
image_width=800
image_height=600
color_channels=3

beta=5#less content ratio
alpha=200#or else try 200

vgg = scipy.io.loadmat("vgg.mat")#loading the weights of the vgg16 model
layers = vgg['layers']#loading the all weights of all the layers

mean_values= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def preprocess_input(path):
    """
    :param path:
    :return:input image-mean values

    This function is used to
    """
    mean_values = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    image = scipy.misc.imread(path)#reading image from path
    image=scipy.misc.imresize(image, (image_height, image_width,color_channels))#reshaping image as (600,800,3)
    image = np.reshape(image, ((1,) + image.shape))#reshaping image as (1,600,800,3)
    image = image - mean_values#subtracting mean values
    return image

def output(path,image):
    """
    This function is used to

    :param path:output directory where image needs to be saved
    :param image: image obtained by neural style transfer
    """
    image = image + mean_values
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def weight(l):
    """
    returns the weights of each of the required layers of VGG19
    """
    W=layers[0][l][0][0][2][0][0]#accessing the weights of given layer l
    b=layers[0][l][0][0][2][0][1]#accessing the biases of given layer l
    return W,b

def relu_func(input):
    """
    :param input:layer
    :return: input layer after applying the rectified linear activation function
    """
    return tf.nn.relu(input)

def conv2d(prev,l):
    """
    function performs convolution of the inputted layer using weights of the VGG16 layer
    :param prev: layer of which the convolution has to be performed
    :param l: layer of the VGG16 network whose weights and biases need to be accessed
    :return: convoluted layer
    """
    W=weight(l)[0]#accessing weights of layer l of VGG
    b=weight(l)[1]#accessing biases of layer l of VGG16
    W=tf.constant(W)
    B= tf.constant(np.reshape(b, (b.size)))
    return tf.nn.conv2d(input=prev,filter=W,strides=[1,1,1,1],padding="SAME")+B

def relu_plus_conv(prev,l):
    """
    performs relu and convolution together
    :param prev: layer for the function to perform relu+conv on
    :param l: layer of VGG16 with appropriate weigths and biases to be accessed
    :return: returns the relu+conv layer
    """
    return relu_func(conv2d(prev,l))

def pooling(prev):
    """
    performs average pooling of the prev layer(average pooling gives better results as compared to max pooling)
    :param prev:
    :return:
    """
    return tf.nn.avg_pool(prev,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def build():
    graph = {}
    graph['input'] = graph['input'] = tf.Variable(np.zeros((1, image_height, image_width, color_channels)), name="in", dtype=tf.float32)
    graph['conv1_1'] = relu_plus_conv(graph['input'], 0)
    graph['conv1_2'] = relu_plus_conv(graph['conv1_1'], 2)
    graph['pool1'] = pooling(graph['conv1_1'])

    graph['conv2_1'] = relu_plus_conv(graph['pool1'], 5)
    graph['conv2_2'] = relu_plus_conv(graph['conv2_1'], 7)
    graph['pool2'] = pooling(graph['conv2_2'])

    graph['conv3_1'] = relu_plus_conv(graph['pool2'], 10)
    graph['conv3_2'] = relu_plus_conv(graph['conv3_1'], 12)
    graph['conv3_3'] = relu_plus_conv(graph['conv3_2'], 14)
    graph['conv3_4'] = relu_plus_conv(graph['conv3_3'], 16)
    graph['pool3'] = pooling(graph['conv3_4'])

    graph['conv4_1'] = relu_plus_conv(graph['pool3'], 19)
    graph['conv4_2'] = relu_plus_conv(graph['conv4_1'], 21)
    graph['conv4_3'] = relu_plus_conv(graph['conv4_2'], 23)
    graph['conv4_4'] = relu_plus_conv(graph['conv4_3'], 25)
    graph['pool4'] = pooling(graph['conv4_4'])

    graph['conv5_1'] = relu_plus_conv(graph['pool4'], 28)
    graph['conv5_2'] = relu_plus_conv(graph['conv5_1'], 30)
    graph['conv5_3'] = relu_plus_conv(graph['conv5_2'], 32)
    graph['conv5_4'] = relu_plus_conv(graph['conv5_3'], 34)
    graph['pool5'] = pooling(graph['conv5_4'])
    return graph

def generate_noise_image(content_image,):
    """
    adding noise to the content_image, which is then updating by backpropagation
    :param content_image:
    :return: content_image with noise
    """
    noise_image = np.random.uniform(-20, 20,(1, image_height,image_width,color_channels)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * 0.4 + content_image * 0.6#probability
    return input_image

def content_loss(p,x):
    """
    function to measure content loss of the output image. measured over the conv4_2 layer of the output layer
    :param p:
    :param x:
    :return:
    """
    m = p.shape[1] * p.shape[2]#width*height
    n = p.shape[3]#colour channel dimensions
    return (1/(2*n*m))*tf.reduce_sum(tf.pow(x-p,2))#removed 4*n*m

def gram_matrix(n,m,x):
    """
    calculating gram matrix by reshaping x in terms of m and n
    :param n:
    :param m:
    :param x:
    :return:gram matrix
    """
    x = tf.reshape(x, (m, n))
    return tf.matmul(tf.transpose(x), x)#finding aggregate

def style_loss_single(a,g):
    """
    function calculates style loss of content_image for a given layer of the VGG16 network
    :param a:
    :param g:
    :return:
    """
    m=a.shape[1]*a.shape[2]
    n=a.shape[3]
    A=gram_matrix(n,m,a)
    G=gram_matrix(n,m,g)

    return (1/(4*n**2*m**2))*tf.reduce_sum(tf.pow(A-G,2))

def total_style_loss():
    """
    calculates style loss over all layers of given VGG16 network. sums over all losses and finds the net style_loss
    :return:
    """
    style_layer=["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]
    sum=0
    weight=[0.2,0.2,0.2,0.2,0.2]#according to the paper, all layers are weighted accordingly
    for i in range(0,len(style_layer)) :
        sum+=weight[i]*style_loss_single(sess.run(graph[style_layer[i]]), graph[style_layer[i]])
    return sum

def main():
    #build the model
    sess = tf.InteractiveSession()
    content_image = preprocess_input(content_image)


    style=preprocess_input(image_for_style)
    graph= model()

    input_image = generate_noise_image(content_image)

    sess.run(tf.global_variables_initializer())


    sess.run(graph['input'].assign(content_image))
    c_l=content_loss(sess.run(graph['conv4_2']), graph['conv4_2'])


    sess.run(graph['input'].assign(style))
    s_l=total_style_loss()
    """
    sess.run(graph['input'].assign(content_image))
    d_l=diff_loss()
    """
    beta=5#less content ratio
    alpha=200#or else try 200
    total_loss=beta*c_l +alpha*s_l
    #total_loss = beta * c_l + alpha * s_l + l*d_l

    optimizer = tf.train.AdamOptimizer(6.0,0.9,0.999,1e-8)
    train_step = optimizer.minimize(total_loss)
    """
    optimizer = tf.train.AdamOptimizer(10,0.9,0.999,1e-8)
    train_step = optimizer.minimize(total_loss)
    """
    sess.run(tf.global_variables_initializer())

    sess.run(graph['input'].assign(input_image))
    iter=1000
    for it in range(iter):
        sess.run(train_step)
    mixed_image=sess.run(graph['input'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filename='output.jpg'
    output(filename,mixed_image)

main()
