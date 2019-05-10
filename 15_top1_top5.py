import tensorflow as tf
import os


TRAIN_FILE = './train1000.tfrecords'
TEST_FILE = './test1000.tfrecords'
SAVE_PATH = 'D:/download/spyder/save15/checkpoint_train.ckpt'
SAVE_PATH_DIR = 'D:/download/spyder/save15'
BATCH_SIZE = 50
TEST_BATCH_SIZE = 10

TRAINING_SET_SIZE = 24536
TEST_SET_SIZE = 5942
IMAGE_SIZE = 224
CATEGORY = 16

Kind=[ "null","buttercup", "daisy" , "dandelion" , "hibiscus" ,"hollyhock" , "hyacinth" , "inulaJaponica" ,"lilyvalley", "lotus" ,"petunia" ,"pleniflora", "rose" , "sunflower" , "tigerlily" ,"tulip" ]

L = [0]*CATEGORY
W = [0]*CATEGORY
S = [0] * CATEGORY
# global_step=100

def read_tfrecords(tfrecords_file, is_training=True):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    if (is_training):
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        label = tf.cast(features['train/label'], tf.int64)
        image = tf.reshape(image, [224, 224, 3])

        images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=100,
                                                        min_after_dequeue=40)


    else:
        feature = {'test/image': tf.FixedLenFeature([], tf.string),
                   'test/label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['test/image'], tf.float32)
        label = tf.cast(features['test/label'], tf.int64)
        image = tf.reshape(image, [224, 224, 3])
        images_batch, labels_batch = tf.train.batch([image, label], batch_size=TEST_BATCH_SIZE)

    return images_batch, labels_batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.03)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.2, shape=shape)
    return tf.Variable(initial)


def max_pool_2x2(input_tensor, layer_name, kszie=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    pool = tf.nn.max_pool(input_tensor, ksize=kszie, strides=strides, padding='SAME')
    return pool


def avg_pool(input_tensor, layer_name, ksize, strides ,padding='VALID'):
    pool = tf.nn.avg_pool(input_tensor, ksize=ksize, strides=strides, padding=padding)
    return pool


def conv2d(input_tensor, kernel_dim, layer_name, strides=[1, 1, 1, 1], act=tf.nn.relu, prob=True):
    weights = weight_variable(kernel_dim)
    biases = bias_variable([kernel_dim[-1]])
    preactivate = tf.nn.conv2d(input_tensor, weights, strides=strides, padding='SAME') + biases
    batch_normalization = tf.layers.batch_normalization(preactivate, training=prob)
    activations = act(batch_normalization, name='activation')
    return activations


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    weights = weight_variable([input_dim, output_dim])
    biases = bias_variable([output_dim])
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act(preactivate, name='activation')
    return activations



def dropout(input_tensor, keep_prob, layer_name):
    drop = tf.nn.dropout(input_tensor, keep_prob)
    return drop


def myModule(input_tensor, input_dim, k1, k2, k3, k4, k5, k6, layer_name, prob,s=1):

    L1 = conv2d(input_tensor, [1, 1, input_dim, k1], layer_name + '_1', prob=prob,strides=[1,s,s,1])
    L2_1 = conv2d(input_tensor, [1, 1, input_dim, k2], layer_name + '_2_1', prob=prob)
    L2_2 = conv2d(L2_1, [3, 3, k2, k3], layer_name + '_2_2', prob=prob,strides=[1,s,s,1])
    L3_1 = conv2d(input_tensor, [1, 1, input_dim, k4], layer_name + '_3_1', prob=prob)
    L3_2 = conv2d(L3_1, [5, 5, k4, k5], layer_name + '_3_2', prob=prob,strides=[1,s,s,1])
    L4_pool = max_pool_2x2(input_tensor, layer_name + '_4_1', [1, 3, 3, 1], [1, 1, 1, 1])
    L4 = conv2d(L4_pool, [1, 1, input_dim, k6], layer_name + '_4_2', prob=prob,strides=[1,s,s,1])

    return tf.concat([L1, L2_2, L3_2, L4], 3), k1 + k3 + k5 + k6


def myCNN(x, keep_prob, prob):
    L1 = conv2d(x, [7, 7, 3, 64], 's_layer1', [1, 2, 2, 1], prob=prob)
    
    L1_pool = max_pool_2x2(L1, 'pool1', [1, 2, 2, 1])
    
    L2 = conv2d(L1_pool, [3, 3, 64, 64], 's_layer2', prob=prob)

    L3 = conv2d(L2, [3, 3, 64, 192], 's_layer3', prob=prob)
    L3_pool = max_pool_2x2(L3, 'pool2', [1, 2, 2, 1])

    
    form1, form1_out_dim = myModule(L3_pool, 192, 64, 96, 128, 16, 32, 32, 'form1', prob=prob,s=1)
    
    form2, form2_out_dim = myModule(form1, form1_out_dim, 128, 128, 192, 32, 96, 64, 'form2', prob=prob,s=1)

    form2_pool, form2_pool_out_dim =  myModule(form2, form2_out_dim, 128, 128, 192, 32, 96, 64, 'form2_pool', prob=prob,s=2)

    form3, form3_out_dim = myModule(form2_pool, form2_pool_out_dim, 192, 96, 208, 16, 48, 64, 'form3', prob=prob,s=1)
          
    form6_pool,form6_pool_out_dim =  myModule(form3, form3_out_dim, 160, 112, 224, 24, 64, 64, 'form6_pool', prob=prob,s=2)
           
    form7_avg_pool=avg_pool(form6_pool,'avg_pool',[1,7,7,1],[1,1,1,1])
    drop_out = dropout(form7_avg_pool, keep_prob, 'form7_avg_pool_dropout')
    
    L34_avg_pool_flat = tf.reshape(drop_out, [-1, 1*1*512])
    
    fc1= nn_layer(L34_avg_pool_flat,512,CATEGORY,'fc1')
    
    return fc1

def evaluation(softmax, y, k):
    
    for i in range(TEST_BATCH_SIZE):
        x=[]
        
        if k==1:
            S[y[i]]+=1

        for j in range(CATEGORY):
            x.append([j,softmax[0][i][j]])
            
        # array[[CATEGORY_SIZE],[],[]... test_batch_size]

        for a in range(CATEGORY-1):
            for b in range(a+1,CATEGORY):
                if( x[a][1]<x[b][1]):
                    temp=x[a]
                    x[a]=x[b]
                    x[b]=temp                    

        for n in range(k):
            if x[n][0]== y[i]:
                if(k==1):
                    L[y[i]]+=1
                if(k==5):
                    W[y[i]]+=1

                    
                    

def train_eval():
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

    keep_prob = tf.placeholder(tf.float32)
    prob = tf.placeholder(tf.bool)

    logits_out= myCNN(x, keep_prob, prob)

    test_batch_x, test_batch_y = read_tfrecords(TEST_FILE,False)
    
    softmax=tf.to_float(tf.nn.softmax(logits_out))

    

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(SAVE_PATH_DIR)
    with tf.Session() as sess:

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, os.path.join(SAVE_PATH))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        total_batch = int(TEST_SET_SIZE / TEST_BATCH_SIZE)

        for j in range(total_batch):
            test_batch_x_out, test_batch_y_out = sess.run([test_batch_x, test_batch_y])
            accuracy= sess.run([softmax], feed_dict={x: test_batch_x_out, keep_prob: 1.0, prob: False})
            evaluation(accuracy, test_batch_y_out, 1)
            evaluation(accuracy, test_batch_y_out, 5)
            
        for i in range(1,CATEGORY):
            print(L[i],W[i],S[i])
        for i in range(1,CATEGORY):
            print(Kind[i],L[i]/S[i],W[i]/S[i])
        
        #print('Accuracy= {:.3f}'.format(total_acc / total_batch))
        A=0
        B=0
        for i in range(1,CATEGORY):
                A+=L[i]
                B+=S[i]
            
            
        print(B)
        print('Accuracy= {:.3f}'.format(A / B))
        
        coord.request_stop()
        coord.join(threads)
        sess.close()


train_eval()