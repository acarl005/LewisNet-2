import sys
import tensorflow as tf
import numpy as np
from matplotlib import patches as patches
from matplotlib import pyplot as plt


from alexnet import AlexNet
from image_loader import load_images

files = sys.argv[1:]
imgs = load_images(files)

m, h, w, _ = imgs.shape

# load the saved tensorflow model and evaluate a list of paths to PNG files (must be 150x150)
num_classes = 1

X = tf.placeholder(tf.float32, shape=(m, h, w, 1))
Y = tf.placeholder(tf.float32, shape=(m, num_classes))
dropout = tf.placeholder(tf.float32)

model = AlexNet(X, dropout, num_classes)

predictions = model.logits > 0

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./tensorflow-ckpt/model.ckpt")
    logits = sess.run(model.logits, feed_dict={X: imgs, dropout: 0})
    _, y_h, y_w, __ = logits.shape
    width_between = (w - 150) / y_w
    height_between = (h - 150) / y_h
    for m, res in enumerate(logits):
        fig, ax = plt.subplots(1, figsize=(8.5, 11))
        ax.imshow(np.squeeze(imgs[m]), cmap="gray")
        for i, row in enumerate(res):
            print(" ".join([ "%.2f" % num if num > 1.7 else "...." for num in row ]))
            for j, val in enumerate(row):
                if (val > 1.8):
                    h_start = i * height_between
                    w_start = j * width_between
                    rect = patches.Rectangle((w_start, h_start), 150, 150, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        fig.savefig('temp-demo.png')

