import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import data_reader
import my_vgg16
import os
import shutil

#定义FLAGS
tf.app.flags.DEFINE_integer('image_size', 224, 'Needs to provide same value as in traning')
tf.app.flags.DEFINE_integer('batch_size', 200, 'batch size')
tf.app.flags.DEFINE_integer('max_step', 20000, 'the max traning steps')
tf.app.flags.DEFINE_integer('eval_steps', 20, 'the step num to eval')
tf.app.flags.DEFINE_integer('num_readers', 5, 'The number of parallel readers that read data from the dataset')

tf.app.flags.DEFINE_string('train_dir', '../tfRecord/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_dir', '../tfRecord/', 'the test dataset dir')
tf.app.flags.DEFINE_string('checkpoints_dir', '../checkpoints/model.ckpt', 'the checkpoints dir')
tf.app.flags.DEFINE_string('log_path', '../log/tensor_log', 'the tensorBoard dir')

#加载Flags
FLAGS = tf.app.flags.FLAGS

#定义全局变量

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name=name, values=var)

        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev' + name, stddev)

def model_define_vgg16(x_image, keep_prob):

   def conv2d(x, W):
       return tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding='SAME')


   def max_pool_2x2(x):
       return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


   def weight_variable(shape):
       initial = tf.truncated_normal(shape, stddev=0.1)
       return tf.Variable(initial)


   def bias_variable(shape):
       initial = tf.constant(0.1, shape=shape)
       return tf.Variable(initial)

   w_conv1_1 = weight_variable([3, 3, 3, 64])
   b_conv1_1 = bias_variable([64])
   h_conv1_1 = tf.nn.relu(conv2d(x_image, w_conv1_1) + b_conv1_1)

   w_conv1_2 = weight_variable([3, 3, 64, 64])
   b_conv1_2 = bias_variable([64])
   h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, w_conv1_2) + b_conv1_2)
   h_pool1 = max_pool_2x2(h_conv1_2)

   w_conv2_1 = weight_variable([3, 3, 64, 128])
   b_conv2_1 = bias_variable([128])
   h_conv2_1 = tf.nn.relu(conv2d(h_pool1, w_conv2_1) + b_conv2_1)

   w_conv2_2 = weight_variable([3, 3, 128, 128])
   b_conv2_2 = bias_variable([128])
   h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, w_conv2_2) + b_conv2_2)
   h_pool2 = max_pool_2x2(h_conv2_2)

   w_conv3_1 = weight_variable([3, 3, 128, 256])
   b_conv3_1 = bias_variable([256])
   h_conv3_1 = tf.nn.relu(conv2d(h_pool2, w_conv3_1) + b_conv3_1)

   w_conv3_2 = weight_variable([3, 3, 256, 256])
   b_conv3_2 = bias_variable([256])
   h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, w_conv3_2) + b_conv3_2)

   w_conv3_3 = weight_variable([3, 3, 256, 256])
   b_conv3_3 = bias_variable([256])
   h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, w_conv3_3) + b_conv3_3)
   h_pool3 = max_pool_2x2(h_conv3_3)

   w_conv4_1 = weight_variable([3, 3, 256, 512])
   b_conv4_1 = bias_variable([512])
   h_conv4_1 = tf.nn.relu(conv2d(h_pool3, w_conv4_1) + b_conv4_1)

   w_conv4_2 = weight_variable([3, 3, 512, 512])
   b_conv4_2 = bias_variable([512])
   h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, w_conv4_2) + b_conv4_2)

   w_conv4_3 = weight_variable([3, 3, 512, 512])
   b_conv4_3 = bias_variable([512])
   h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, w_conv4_3) + b_conv4_3)
   h_pool4 = max_pool_2x2(h_conv4_3)

   w_conv5_1 = weight_variable([3, 3, 512, 512])
   b_conv5_1 = bias_variable([512])
   h_conv5_1 = tf.nn.relu(conv2d(h_pool4, w_conv5_1) + b_conv5_1)

   w_conv5_2 = weight_variable([3, 3, 512, 512])
   b_conv5_2 = bias_variable([512])
   h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, w_conv5_2) + b_conv5_2)

   w_conv5_3 = weight_variable([3, 3, 512, 512])
   b_conv5_3 = bias_variable([512])
   h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, w_conv5_3) + b_conv5_3)
   h_pool5 = max_pool_2x2(h_conv5_3)

   shape = int(np.prod(h_pool5.get_shape()[1:]))
   W_fc1 = weight_variable([shape, 4096])
   b_fc1 = bias_variable([4096])
   h_pool2_flat = tf.reshape(h_pool5, [-1, shape])
   h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

   W_fc2 = weight_variable([4096, 48])
   b_fc2 = bias_variable([48])

   logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

   return logits, [w_conv1_1, b_conv1_1, w_conv1_2, b_conv1_2, w_conv2_1, b_conv2_1, w_conv2_2, b_conv2_2, w_conv3_1, w_conv3_1, w_conv3_2, w_conv3_2,
                   w_conv3_3, b_conv3_3, w_conv4_1, b_conv4_1, w_conv4_2, b_conv4_2, w_conv4_3, b_conv4_3, w_conv5_1, b_conv5_1, w_conv5_2, b_conv5_2,
                   w_conv5_3, b_conv5_3, W_fc1, b_fc1, W_fc2, b_fc2]


def model_train():
    #预定义占位符
    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
        input_labels = tf.placeholder(tf.int64, [FLAGS.batch_size])
        input_validate_images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
        input_test_images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])

    with tf.name_scope('input_image'):
        image_reshpe = tf.reshape(input_images, [-1, FLAGS.image_size, FLAGS.image_size, 3])
        tf.summary.image('input', image_reshpe, 10)

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        global_step = slim.train.create_global_step()

    #读取TFRecord
    data_set = data_reader.get_split('main_train', FLAGS.train_dir)
    proider = slim.dataset_data_provider.DatasetDataProvider(
        data_set,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.image_size,
        common_queue_min=10 * FLAGS.batch_size,
    )

    [images, labels] = proider.get(['image', 'label'])

    #图像增强
    images = tf.image.convert_image_dtype(images, tf.int32)
    images = tf.image.random_brightness(images, max_delta=0.3)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)

    #调用 tf.train.batch 来获取batch 大小的数据流
    image_batch, label_batch = tf.train.shuffle_batch(
        [images, labels],
        batch_size=FLAGS.batch_size,
        num_threads=4,
        capacity=5 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.batch_size
    )

    with tf.name_scope('net'):
        # logits = my_vgg16.inference_op(image_batch, keep_prob)
        logits, params = model_define_vgg16(image_batch, keep_prob)
        # tf.summary.scalar('logits', logits)

    #定义loss
    with tf.name_scope('loss_function'):
        # cross_entropy = tf.nn.(logits=logits, labels=label_batch)
        # loss = tf.log(tf.clip_by_value(cross_entropy, 1e-8, tf.reduce_max(cross_entropy)))
        # cross_entropy = -tf.reduce_sum(labels * tf.log(label_output))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels)
        loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('loss', loss)

    with tf.name_scope('train_step'):
        # 学习率
        learning_rate = tf.train.exponential_decay(
            0.001, global_step, 200, 0.98
        )
        # 优化器
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # 训练步骤
        # train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('train_accuracy'):
        # 定义准确率
        # logits_batch = tf.argmax(label_output, 1)
        logits_label = tf.argmax(logits, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(logits_label, input_labels), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 验证数据集
    data_set = data_reader.get_split('main_validate', FLAGS.train_dir)
    proider = slim.dataset_data_provider.DatasetDataProvider(
        data_set,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.image_size,
        common_queue_min=10 * FLAGS.batch_size,
    )


    [validate_images, validate_labels] = proider.get(['image', 'label'])
    validate_images = tf.image.resize_images(validate_images, new_size)

    # 调用 tf.train.batch 来获取batch 大小的数据流
    validate_image_batch, validate_label_batch = tf.train.batch(
        [validate_images, validate_labels],
        batch_size=FLAGS.batch_size,
        num_threads=4,
        capacity=5 * FLAGS.batch_size
    )

    saver = tf.train.Saver()

    # 整理所有定义好的日志生成操作（summary.scalar,  histogram), 在sess。run中执行可以将所有文件写入日志
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # 初始化tensor日志，并将计算图写入日志
        # if (os.path.exists(FLAGS.log_path)):
        #     shutil.rmtree(FLAGS.log_path)

        writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

        for index in range(FLAGS.max_step):
            train_images, train_labels = sess.run([image_batch, label_batch])
            train_dic = {keep_prob: 0.8,
                         input_images: train_images,
                         input_labels: train_labels}
            summary, train, train_logits, = sess.run([merged, train_op, logits], feed_dict=train_dic)

            # 将所有日志写入文件
            writer.add_summary(summary, index)
            if index % FLAGS.eval_steps == 0:
                #配置运行时需要记录的信息
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                validate_images, validate_labels = sess.run([validate_image_batch, validate_label_batch])
                validate_dic = {keep_prob: 0.8,
                                input_validate_images: validate_images,
                                input_labels: validate_labels}
                validate_acc, validate_loss = sess.run([accuracy, loss], feed_dict=validate_dic)
                rate = sess.run(learning_rate)

                print('After %d traing setps, logits_label is %s' % (index, train_logits))
                print('After %d traing setps, labels is %s' % (index, train_labels))
                print('After %d traing setps, accuracy is %g' % (index, validate_acc))
                print('After %d traing setps, loss is %g' % (index, validate_loss))
                print('After %d traing setps, learning_rate is %g' % (index, rate))

            if index % 1000 == 0:
                saver.save(sess, FLAGS.checkpoints_dir, global_step=global_step)

        coord.join(threads)

    writer.close()

def test_model():
    saver = tf.train.saver()


if __name__ == '__main__':
    model_train()