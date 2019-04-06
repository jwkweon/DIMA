import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
#X_dim = mnist.train.images.shape[1]
#y_dim = mnist.train.labels.shape[1]
h_dim = 128

def load_datas():
    data = np.array(glob('./in/*.jpg'))
    sample = [imread(sample_file) for sample_file in data]
    sample_images = np.array(sample).astype(np.float32)

    return data, sample_images

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

'''
initializer = tf.random_normal_initializer(0., 0.02)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
model.add(tf.keras.layers.LeakyReLU())
'''


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Cond = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

D_W1 = tf.Variable(tf.random_normal([3, 3, 6, 32], stddev=0.01))    #16 * 16 * 32
D_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))   #8 * 8 * 64
D_W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))  #4 * 4 * 128
D_W4 = tf.Variable(tf.random_normal([4 * 4 * 128, 512], stddev=0.01))  #FC 512
D_W5 = tf.Variable(tf.random_normal([512, 1], stddev=0.01))  #FC 1

theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5]

#x = 28*28*3 (fake/real) y = 28*28*3 (color)
def discriminator(x, y):
    inputs = tf.concat(axis=2, values=[x, y])
    D_L1 = tf.nn.relu(tf.nn.conv2d(inputs, D_W1,                       # l1a shape=(?, 32, 32, 32)
                    strides=[1, 1, 1, 1], padding='SAME'))
    D_L1 = tf.nn.max_pool(D_L1, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 32)
                    strides=[1, 2, 2, 1], padding='SAME')

    D_L2 = tf.nn.relu(tf.nn.conv2d(D_L1, D_W2,                       # l1a shape=(?, 16, 16, 64)
                    strides=[1, 1, 1, 1], padding='SAME'))
    D_L2 = tf.nn.max_pool(D_L2, ksize=[1, 2, 2, 1],              # l1 shape=(?, 8, 8, 64)
                    strides=[1, 2, 2, 1], padding='SAME')

    D_L3 = tf.nn.relu(tf.nn.conv2d(D_L2, D_W3,                       # l1a shape=(?, 8, 8, 128)
                    strides=[1, 1, 1, 1], padding='SAME'))
    D_L3 = tf.nn.max_pool(D_L3, ksize=[1, 2, 2, 1],              # l1 shape=(?, 4, 4, 128)
                    strides=[1, 2, 2, 1], padding='SAME')
    D_L3 = tf.reshape(D_L3, [-1, D_W4.get_shape().as_list()[0]])    # ? 2048

    D_L4 = tf.nn.relu(tf.matmul(D_L3, D_W4))

    D_L5 = tf.matmul(D_L4, D_W5)

    return D_L5
    #D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    #D_logit = tf.matmul(D_h1, D_W2) + D_b2
    #D_prob = tf.nn.sigmoid(D_logit)

    #return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

G_W1 = tf.Variable(tf.random_normal([3, 3, 4, 32], stddev=0.01))    #16 * 16 * 32
G_W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))   #8 * 8 * 64
G_W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))  #4 * 4 * 128
G_W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))  #2 * 2 * 256
G_W5 = tf.Variable(tf.random_normal([3, 3, 256, 128], stddev=0.01))  #4 * 4 * 128
G_W6 = tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.01))   #8 * 8 * 64
G_W7 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01))    #16 * 16 * 32
G_W8 = tf.Variable(tf.random_normal([3, 3, 32, 3], stddev=0.01))    #32 * 32 * 3

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_W7, G_W8]


def generator(z, y):
    inputs = tf.concat(axis=2, values=[z, y])
    G_L1 = tf.nn.relu(tf.nn.conv2d(inputs, G_W1,                       # l1a shape=(?, 32, 32, 32)
                    strides=[1, 1, 1, 1], padding='SAME'))
    G_L1 = tf.nn.max_pool(G_L1, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 32)
                    strides=[1, 2, 2, 1], padding='SAME')

    G_L2 = tf.nn.relu(tf.nn.conv2d(G_L1, G_W2,                       # l1a shape=(?, 16, 16, 64)
                    strides=[1, 1, 1, 1], padding='SAME'))
    G_L2 = tf.nn.max_pool(G_L2, ksize=[1, 2, 2, 1],              # l1 shape=(?, 8, 8, 64)
                    strides=[1, 2, 2, 1], padding='SAME')

    G_L3 = tf.nn.relu(tf.nn.conv2d(G_L2, G_W3,                       # l1a shape=(?, 8, 8, 128)
                    strides=[1, 1, 1, 1], padding='SAME'))
    G_L3 = tf.nn.max_pool(G_L3, ksize=[1, 2, 2, 1],              # l1 shape=(?, 4, 4, 128)
                    strides=[1, 2, 2, 1], padding='SAME')

    G_L4 = tf.nn.relu(tf.nn.conv2d(G_L3, G_W4,                       # l1a shape=(?, 4, 4, 256)
                    strides=[1, 1, 1, 1], padding='SAME'))
    G_L4 = tf.nn.max_pool(G_L4, ksize=[1, 2, 2, 1],              # l1 shape=(?, 2, 2, 256)
                    strides=[1, 2, 2, 1], padding='SAME')

    G_L5 = tf.nn.relu(tf.nn.conv2d_transpose(G_L4, G_W5,             # l1a shape=(?, 4, 4, 128)
                    output_shape=[-1, 4, 4, 128], strides=[1, 2, 2, 1], padding='SAME'))

    G_L6 = tf.nn.relu(tf.nn.conv2d_transpose(G_L5, G_W6,             # l1a shape=(?, 8, 8, 64)
                    output_shape=[-1, 8, 8, 64], strides=[1, 2, 2, 1], padding='SAME'))

    G_L7 = tf.nn.relu(tf.nn.conv2d_transpose(G_L6, G_W7,             # l1a shape=(?, 4, 4, 32)
                    output_shape=[-1, 16, 16, 32], strides=[1, 2, 2, 1], padding='SAME'))

    G_L8 = tf.nn.relu(tf.nn.conv2d_transpose(G_L7, G_W8,             # l1a shape=(?, 4, 4, 32)
                    output_shape=[-1, 32, 32, 3], strides=[1, 2, 2, 1], padding='SAME'))

    return G_L8

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real = discriminator(X, y)
D_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out5/'):
    os.makedirs('out5/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        n_sample = 16

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 5] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        plt.savefig('out5/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, y_mb = mnist.train.next_batch(mb_size)

    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
