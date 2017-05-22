from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.eval_tools.pycocotools.coco import COCO
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/home/meteorshub/code/im2txt/im2txt/model/train",
                       "Model checkpoint file or directory containing a "
                       "model-backup checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/home/meteorshub/code/im2txt/im2txt/data/mscoco/word_counts.txt",
                       "Text file containing the vocabulary.")
tf.flags.DEFINE_string("image_dir",
                       "/media/meteorshub/resource/dataset/mscoco/images/val2014/",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("annotation_file",
                       "/media/meteorshub/resource/dataset/mscoco/annotations/captions_val2014.json",
                       "annotations file for COCO api")
tf.flags.DEFINE_string("result_file",
                       "/home/meteorshub/code/im2txt/im2txt/eval_tools/results/captions_val2014_meteorshub_results.json",
                       "result file path")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    result = []

    with tf.Session(graph=g) as sess:
        # Load the model-backup from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        coco = COCO(FLAGS.annotation_file)
        image_ids = coco.getImgIds()
        img_objs = coco.loadImgs(image_ids)

        for i in range(len(img_objs)):
            with tf.gfile.GFile(os.path.join(FLAGS.image_dir, img_objs[i]['file_name']), "r") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)

            sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
            sentence = " ".join(sentence)
            result.append({"image_id": img_objs[i]['id'], "caption": sentence})
            if (i + 1) % 10 == 0:
                print("Captions for image %s in %s done" % (i + 1, len(img_objs)))

            if i > 1000:
                break

    json.dump(result, open(FLAGS.result_file, 'w'))


if __name__ == "__main__":
    tf.app.run()
