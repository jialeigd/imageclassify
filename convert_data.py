# coding=utf-8
"""
数据转换,将数据从原始格式转化为TFRecord格式提供给图像分类的主程序使用。
这里处理的数据原始格式为一个根目录下的多个子目录，每个子目录的文件名为
图片类别名称，子目录下的图片都属于该类。

传入参数:
    --dataset_name: 数据集名称
    --train_fraction: 训练集比例
    --test_fraction: 测试集比例

输出:
    1. dataset目录: 包含按照shard数量拆分后的train, validation 和 test的TFRecord格式的数据集
    2. counts.txt: 拆分后train, validation 和 test数据集包含的数据量
    3. main_labels_to_class.txt: 标签与数字类别的对应关系
"""

import math
import os
import random

import tensorflow as tf
import data_utils
import argparse

# 定义一些全局常量
# TRAIN_FRACTION: 定义训练数据比例
TRAIN_FRACTION = 0.9

# TEST_FRACTION: 定义测试数据比例
TEST_FRACTION = 0.1

# 定义随机数种子值
_RANDOM_SEED = 0

# 定义SHARD数量
_NUM_SHARDS = 10

# DATASET_SUBDIR: 定义数据子文件名
DATASET_INPUT = "../cropImage"
DATASET_OUTPUT = "../tfRecord"

# 定义任务名称
TASKS = ["main"]

split_to_count = {}


# 获取全部文件名和数据标签
def _get_all_filenames(dataset_dir, class_label):
    filenames_and_labels = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            filenames_and_labels.extend(_get_all_filenames(path, class_label))
        elif data_utils.is_picture_file(filename):
            filenames_and_labels.append((path, class_label))
    return filenames_and_labels


# 获取转换为TFRecord类型的数据文件名
def _get_dataset_filename(dataset_dir, split_name, task_name, shard_id):
    output_filename = '%s/%s_%s_%05d-of-%05d.tfrecord' % (
        DATASET_OUTPUT, task_name, split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


# 数据转换，读取图像数据并将数据转换成TFRecord格式
def _convert_dataset(split_name, task_name, filenames_and_labels, dataset_dir):
    num_per_shard = int(math.ceil(len(filenames_and_labels) / float(_NUM_SHARDS)))
    print ("For task %s, split %s, read data and convert into %d shards with %d examples per shard" % (
        task_name, split_name, _NUM_SHARDS, num_per_shard))
        
    with tf.Graph().as_default():
        image_reader = data_utils.ImageReader()
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, task_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    print ("Start reading images in shard %d" % shard_id)
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames_and_labels))
                    for i in range(start_ndx, end_ndx):
                        file_name = filenames_and_labels[i][0];
                        image_data = tf.gfile.FastGFile(file_name, 'rb').read()
                        if (image_data is None):
                            continue

                        print("example fimename is :" + filenames_and_labels[i][0])
                        height, width = image_reader.read_image_dims(sess, image_data)
                        class_id = filenames_and_labels[i][1]

                        example = data_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id, filenames_and_labels[i][0])
                        tfrecord_writer.write(example.SerializeToString())


# 输出数据集
def _output_dataset(filenames_and_labels, labels_to_class_names, task_name, dataset_dir):
    random.shuffle(filenames_and_labels)
    all_examples = len(filenames_and_labels)
    train_examples = int(all_examples * TRAIN_FRACTION)
    test_examples = int(all_examples * TEST_FRACTION)
    validate_examples = all_examples - train_examples - test_examples
    print ("Generated %d training examples, %d testing examples and %d validation examples for the main task. " % (
        train_examples, test_examples, validate_examples))
      
    _convert_dataset('test', task_name, filenames_and_labels[:test_examples], dataset_dir)
    _convert_dataset('train', task_name, filenames_and_labels[test_examples:train_examples + test_examples], dataset_dir)
    _convert_dataset('validate', task_name, filenames_and_labels[train_examples + test_examples:], dataset_dir)

    data_utils.write_label_file(
        labels_to_class_names, dataset_dir, "%s_labels_to_class.txt" % task_name)

    global split_to_count
    split_to_count["%s_train" % task_name] = train_examples
    split_to_count["%s_test" % task_name] = test_examples
    split_to_count["%s_validate" % task_name] = validate_examples


# 主要数据转换函数，将输入的数据集转换后存储到 output_dir
def _process_main(dataset_dir, output_dir):
    labels_to_class_names = {}
    filenames_and_labels = []
    cur_id = 0
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            filenames_and_labels.extend(_get_all_filenames(path, cur_id))
            labels_to_class_names[cur_id] = filename
            cur_id += 1
    
    _output_dataset(filenames_and_labels, labels_to_class_names, "main", dataset_dir)


# 处理数据标签
def _process_labels(dataset_dir, output_dir, task_name, cur_dict):
    labels_to_class_names = {}
    filenames_and_labels = []
    cur_id = 0
    for fir_dir_name in os.listdir(dataset_dir):
        if fir_dir_name == DATASET_OUTPUT:
            continue
        if not fir_dir_name in cur_dict:
            continue
        path = os.path.join(dataset_dir, fir_dir_name)
        if not os.path.isdir(path):
            continue
               
        for sec_dir_name in os.listdir(path):
            parts = sec_dir_name.split("-")
            if not parts[1] in cur_dict[fir_dir_name]:
                continue
            subdir = os.path.join(path, sec_dir_name)
            if not os.path.isdir(subdir):
                continue
                    
            filenames_and_labels.extend(_get_all_filenames(subdir, cur_id))
            labels_to_class_names[cur_id] = sec_dir_name
            cur_id += 1
    
    _output_dataset(filenames_and_labels, labels_to_class_names, task_name, dataset_dir)


# 运行函数，获取数据文件地址，转换数据
def run():
    if not tf.gfile.Exists(DATASET_INPUT):
        raise("Dataset does not exist.")

    if os.path.exists(DATASET_OUTPUT):
        tf.gfile.DeleteRecursively(DATASET_OUTPUT)

    os.mkdir(DATASET_OUTPUT)

    random.seed(_RANDOM_SEED)
    _process_main(DATASET_INPUT, DATASET_OUTPUT)

    global split_to_count
    data_utils.write_count_file(split_to_count)

    print('\nFinished converting the dataset!')


# 定义传入参数
def parse_args():
    parser = argparse.ArgumentParser(description="Convert data to TFRecord format")
    parser.add_argument('--train_fraction', dest='train_fraction', help='Train Fraction', default='0.6')
    parser.add_argument('--test_fraction', dest='test_fraction', help='Test Fraction', default='0.2')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    TRAIN_FRACTION = float(args.train_fraction)
    TEST_FRACTION = float(args.test_fraction)
    run()

