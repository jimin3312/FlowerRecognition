
import tensorflow  as tf
import numpy as np
import cv2
import sys
import glob
#22879 5813
Shuffle = True  # shuffle the addresses before saving


daisy_train_path = glob.glob('train/daisy/*.jpg')
dandelion_train_path = glob.glob('train/dandelion/*.jpg')
hibiscus_train_path = glob.glob('train/hibiscus/*.jpg')
hollyhock_train_path = glob.glob('train/hollyhock/*.jpg')
lotus_train_path = glob.glob('train/lotus/*.jpg')
petunia_train_path = glob.glob('train/petunia/*.jpg')
pleniflora_train_path = glob.glob('train/pleniflora/*.jpg')
rose_train_path = glob.glob('train/rose/*.jpg')
sunflower_train_path = glob.glob('train/sunflower/*.jpg')
tulip_train_path = glob.glob('train/tulip/*.jpg')
null_train_path = glob.glob('train/null/*.jpg')

buttercup_train_path = glob.glob('train/buttercup/*.jpg')
hyacinth_train_path = glob.glob('train/hyacinth/*.jpg')
inulaJaponica_train_path = glob.glob('train/inulaJaponica/*.jpg')
lilyvalley_train_path = glob.glob('train/lilyvalley/*.jpg')
tigerlily_train_path = glob.glob('train/tigerlily/*.jpg')

def train_val_test(addrs):
    np.random.shuffle(addrs)
    train_addrs = addrs[0:int(0.8 * len(addrs))]
    test_addrs = addrs[int(0.8 * len(addrs)):]

    return train_addrs, test_addrs

daisy_train, daisy_test = train_val_test(daisy_train_path)
dandelion_train, dandelion_test = train_val_test(dandelion_train_path)
hibiscus_train, hibiscus_test = train_val_test(hibiscus_train_path)
hollyhock_train, hollyhock_test = train_val_test(hollyhock_train_path)
lotus_train, lotus_test = train_val_test(lotus_train_path)
petunia_train, petunia_test = train_val_test(petunia_train_path)
pleniflora_train, pleniflora_test = train_val_test(pleniflora_train_path)
rose_train, rose_test = train_val_test(rose_train_path)
sunflower_train, sunflower_test = train_val_test(sunflower_train_path)
tulip_train, tulip_test = train_val_test(tulip_train_path)
null_train, null_test = train_val_test(null_train_path)

buttercup_train, buttercup_test = train_val_test(buttercup_train_path)
hyacinth_train, hyacinth_test = train_val_test(hyacinth_train_path)
inulaJaponica_train, inulaJaponica_test = train_val_test(inulaJaponica_train_path)
lilyvalley_train, lilyvalley_test = train_val_test(lilyvalley_train_path)
tigerlily_train, tigerlily_test = train_val_test(tigerlily_train_path)



# 15종류
Kind=[ "null","buttercup", "daisy" , "dandelion" , "hibiscus" ,"hollyhock" , "hyacinth" , "inulaJaponica" ,"lilyvalley", "lotus" ,"petunia" ,"pleniflora", "rose" , "sunflower" , "tigerlily" ,"tulip" ]


# 15종류
train_addrs = buttercup_train + daisy_train + dandelion_train + hibiscus_train + hollyhock_train + hyacinth_train + inulaJaponica_train + lilyvalley_train + lotus_train + petunia_train + pleniflora_train + rose_train + sunflower_train + tigerlily_train + tulip_train + null_train


# 15종류
test_addrs = buttercup_test + daisy_test + dandelion_test + hibiscus_test + hollyhock_test + hyacinth_test + inulaJaponica_test + lilyvalley_test + lotus_test + petunia_test + pleniflora_test + rose_test + sunflower_test + tigerlily_test + tulip_test


# 15종류
train_labels = [1 if 'buttercup' in a else 2 if 'daisy' in a else 3 if 'dandelion' in a else 4 if 'hibiscus' in a else 5 if 'hollyhock' in a else
               6 if 'hyacinth' in a else 7 if 'inulaJaponica' in a else 8 if 'lilyvalley' in a else 9 if 'lotus' in a else 10 if 'petunia' in a else
               11 if 'pleniflora' in a else 12 if 'rose' in a else 13 if 'sunflower' in a else 14 if 'tigerlily' in a else 15 if 'tulip' in a else 0 for a in train_addrs]

test_labels = [1 if 'buttercup' in a else 2 if 'daisy' in a else 3 if 'dandelion' in a else 4 if 'hibiscus' in a else 5 if 'hollyhock' in a else
               6 if 'hyacinth' in a else 7 if 'inulaJaponica' in a else 8 if 'lilyvalley' in a else 9 if 'lotus' in a else 10 if 'petunia' in a else
               11 if 'pleniflora' in a else 12 if 'rose' in a else 13 if 'sunflower' in a else 14 if 'tigerlily' in a else 15 if 'tulip' in a else 0 for a in test_addrs]

#print(test_labels)
if Shuffle:
    c = list(zip(train_addrs, train_labels))   
    np.random.shuffle(c)
    train_addrs, train_labels = zip(*c)
if Shuffle:
    c = list(zip(test_addrs, test_labels))   
    np.random.shuffle(c)
    test_addrs, test_labels = zip(*c)
def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# =============================================================================
# 
train_filename = 'train1000.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# =============================================================================
# open the TFRecords file
test_filename = 'test1000.tfrecords'  # address to save the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(test_addrs[i])
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()