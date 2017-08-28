import gzip
import os

GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

from scipy import ndimage
from six.moves import urllib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
print ("PACKAGES LOADED")



def CNN(inputs, _is_training=True):
    x   = tf.reshape(inputs, [-1, 28, 28, 1])
    batch_norm_params = {'is_training': _is_training, 'decay': 0.9, 'updates_collections': None}
    net = slim.conv2d(x, 32, [5, 5], padding='SAME'
                     , activation_fn       = tf.nn.relu
                     , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                     , normalizer_fn       = slim.batch_norm
                     , normalizer_params   = batch_norm_params
                     , scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 1024
                    , activation_fn       = tf.nn.relu
                    , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    , normalizer_fn       = slim.batch_norm
                    , normalizer_params   = batch_norm_params
                    , scope='fc4')
    net = slim.dropout(net, keep_prob=0.7, is_training=_is_training, scope='dropout4')  
    out = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return out


# DATA URL
SOURCE_URL      = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY  = "data"
# PARAMETERS FOR MNIST
IMAGE_SIZE      = 28
NUM_CHANNELS    = 1
PIXEL_DEPTH     = 255
NUM_LABELS      = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

# DOWNLOAD MNIST DATA, IF NECESSARY
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

# EXTRACT IMAGES
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH # -0.5~0.5
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data # [image index, y, x, channels]

# EXTRACT LABELS
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# AUGMENT TRAINING DATA
def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        # APPEND ORIGINAL DATA
        expanded_images.append(x)
        expanded_labels.append(y)
        # ASSUME MEDIAN COLOR TO BE BACKGROUND COLOR
        bg_value = np.median(x) # this is regarded as background's value        
        image = np.reshape(x, (-1, 28))

        for i in range(4):
            # ROTATE IMAGE
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            # SHIFT IAMGE
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            # ADD TO THE LIST
            expanded_images.append(np.reshape(new_img_, 784))
            expanded_labels.append(y)
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)
    return expanded_train_total_data

# PREPARE MNIST DATA
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]
    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels



# CONFIGURATION
MODEL_DIRECTORY   = "model/model.ckpt"
LOGS_DIRECTORY    = "logs/train"
training_epochs   = 10
TRAIN_BATCH_SIZE  = 50
display_step      = 500
validation_step   = 500
TEST_BATCH_SIZE   = 5000    


# PREPARE MNIST DATA
batch_size = TRAIN_BATCH_SIZE # BATCH SIZE (50)
num_labels = NUM_LABELS       # NUMBER OF LABELS (10)
train_total_data, train_size, validation_data, validation_labels \
    , test_data, test_labels = prepare_MNIST_data(True)
# PRINT FUNCTION
def print_np(x, str):
    print (" TYPE AND SHAPE OF [%18s ] ARE %s and %14s" 
           % (str, type(x), x.shape,))
print_np(train_total_data, 'train_total_data')
print_np(validation_data, 'validation_data')
print_np(validation_labels, 'validation_labels')
print_np(test_data, 'test_data')
print_np(test_labels, 'test_labels')


# DEFINE MODEL
# PLACEHOLDERS
x  = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10]) #answer
is_training = tf.placeholder(tf.bool, name='MODE')
# CONVOLUTIONAL NEURAL NETWORK MODEL 
y = CNN(x, is_training)
# DEFINE LOSS
with tf.name_scope("LOSS"):
    loss = slim.losses.softmax_cross_entropy(y, y_)
# DEFINE ACCURACY
with tf.name_scope("ACC"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# DEFINE OPTIMIZER
with tf.name_scope("ADAM"):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        1e-4,               # LEARNING_RATE
        batch * batch_size, # GLOBAL_STEP
        train_size,         # DECAY_STEP
        0.95,               # DECAY_RATE
        staircase=True)     # LR = LEARNING_RATE*DECAY_RATE^(GLOBAL_STEP/DECAY_STEP)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)
    # 'batch' IS AUTOMATICALLY UPDATED AS WE CALL 'train_step'

# SUMMARIES
saver = tf.train.Saver()
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', accuracy)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
print ("MODEL DEFINED.")


# OPEN SESSION
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})


# MAXIMUM ACCURACY
max_acc = 0.
# LOOP
for epoch in range(training_epochs): # training_epochs: 10
    # RANDOM SHUFFLE
    np.random.shuffle(train_total_data)
    train_data_   = train_total_data[:, :-num_labels]
    train_labels_ = train_total_data[:, -num_labels:]
    # ITERATIONS
    total_batch = int(train_size / batch_size)
    for iteration in range(total_batch):
        # GET CURRENT MINI-BATCH
        offset = (iteration * batch_size) % (train_size)
        batch_xs = train_data_[offset:(offset + batch_size), :]
        batch_ys = train_labels_[offset:(offset + batch_size), :]
        # OPTIMIZE
        _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op]
                                    , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})
        # WRITE LOG
        summary_writer.add_summary(summary, epoch*total_batch + iteration)

        # DISPLAY
        if iteration % display_step == 0:
            print("Epoch: [%3d/%3d] Batch: [%04d/%04d] Training Acc: %.5f" 
                  % (epoch + 1, training_epochs, iteration, total_batch, train_accuracy))

        # GET ACCURACY FOR THE VALIDATION DATA
        if iteration % validation_step == 0:
            validation_accuracy = sess.run(accuracy,
            feed_dict={x: validation_data, y_: validation_labels, is_training: False})
            print("Epoch: [%3d/%3d] Batch: [%04d/%04d] Validation Acc: %.5f" 
                  % (epoch + 1, training_epochs, iteration, total_batch, validation_accuracy))
        # SAVE THE MODEL WITH HIGEST VALIDATION ACCURACY
        if validation_accuracy > max_acc:
            max_acc = validation_accuracy
            save_path = saver.save(sess, MODEL_DIRECTORY)
            print("  MODEL UPDATED TO [%s] VALIDATION ACC IS %.5f" 
                  % (save_path, validation_accuracy))
print("OPTIMIZATION FINISHED")




# RESTORE SAVED NETWORK
saver.restore(sess, MODEL_DIRECTORY)

# COMPUTE ACCURACY FOR TEST DATA
test_size   = test_labels.shape[0]
total_batch = int(test_size / batch_size)
acc_buffer  = []
for i in range(total_batch):
    offset = (i * batch_size) % (test_size)
    batch_xs = test_data[offset:(offset + batch_size), :]
    batch_ys = test_labels[offset:(offset + batch_size), :]
    y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
    correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
    acc_buffer.append(np.sum(correct_prediction.astype(float)) / batch_size)
print("TEST ACCURACY IS: %.4f" % np.mean(acc_buffer))



