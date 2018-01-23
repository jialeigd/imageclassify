import os
import tensorflow as tf
import data_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_*.tfrecord'
_ITEMS_TO_DESCRIPTIONS = {
    'image' : 'A color image of varying size',
    'label' : 'A single integer between 0 and n-classes',
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded' : tf.FixedLenFeature((), tf.string, default_value = ''),
        'image/format' : tf.FixedLenFeature((), tf.string, default_value=''),
        'image/label' : tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
        'image' : slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label' : slim.tfexample_decoder.Tensor('image/label')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    task_split = split_name.split('_')

    if len(task_split):
        labels_to_names = data_utils.read_label_file("%s_labels_to_class.txt" % task_split[0])
    else:
        labels_to_names = data_utils.read_label_file()

    split_count = data_utils.read_count_file()

    return slim.dataset.Dataset(
        data_sources = file_pattern,
        reader = reader,
        decoder = decoder,
        num_samples = split_count.get(split_name),
        items_to_descriptions = _ITEMS_TO_DESCRIPTIONS,
        num_classes = len(labels_to_names),
        labels_to_names = labels_to_names
    )
