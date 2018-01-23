import os
import tensorflow as tf

PIC_EXTENSIONS = ['.jpg', '.jepg', '.JPG', '.png']

COUNT_FINENAME = 'counts.txt'
LABELS_FILENAME = 'main_labels_to_class.txt'

class ImageReader(object):
    def __init__(self):
        self._decode_jpg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpg = tf.image.decode_jpeg(self._decode_jpg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpg(sess, image_data)
        return image.shape[0], image.shape[1]


    def decode_jpg(self, sess, image_data):
        image = sess.run(self._decode_jpg, feed_dict = {self._decode_jpg_data : image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]

    return tf.train.Feature(int64_list = tf.train.Int64List(value = values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id, filename=""):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded' : bytes_feature(image_data),
        'image/format' : bytes_feature(image_format),
        'image/label' : int64_feature(class_id),
        'image/height'  : int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename' : bytes_feature(filename.encode()),
    }))

def write_label_file(labels_to_class_names, dataset_dir, filename):
    labels_filename = os.path.join(filename)
    with tf.gfile.Open(labels_filename, 'w') as file:
        for k, v in labels_to_class_names.items():
            file.write("%d:%s\r\n" % (k, v))

def read_label_file(filename='LABELS_FILENAME'):
    split_to_count = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            index = line.index(':')
            split_to_count[line[:index]] = int(line[index + 1:])

    return split_to_count

def write_count_file(split_to_count, filename="counts.txt"):

    with tf.gfile.Open(filename, 'w') as file:
        for k, v in split_to_count.items():
            file.write("%s : %d\r\n" % (k, v))

def read_count_file(filename="counts.txt"):
    split_to_count = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            index = line.index(':')
            split_to_count[line[:index].strip()] = int(line[index + 1:].strip())

    return split_to_count


def is_picture_file(file_name):
    for ext in PIC_EXTENSIONS:
        if file_name.endswith(ext):
            return True

    return False