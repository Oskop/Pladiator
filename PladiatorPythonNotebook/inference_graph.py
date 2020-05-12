import tensorflow as tf
from src.models import inception_resnet_v1
import sys
def main():

    traning_checkpoint = 'models/model-20180402-114759.ckpt-275'
    eval_checkpoint = 'model_inference/imagenet_facenet.ckpt'

    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 160, 160, 3])
    output, _ = inception_resnet_v1.inference(data_input, keep_probability=0.8, phase_train=False, bottleneck_layer_size=512)
    label_batch= tf.identity(output, name='label_batch')
    embeddings = tf.identity(output, name='embeddings')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, traning_checkpoint)
        save_path = saver.save(sess, eval_checkpoint)
        print('Model saved in file: %s' % save_path)
if __name__ == '__main__':
    main()
