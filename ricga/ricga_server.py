from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cgi
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

import tensorflow as tf

from ricga import configuration
from ricga import inference_wrapper
from ricga.inference_utils import caption_generator
from ricga.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/home/meteorshub/code/RICGA/ricga/model/train",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/home/meteorshub/code/RICGA/ricga/data/mscoco/word_counts.txt",
                       "Text file containing the vocabulary.")

tf.flags.DEFINE_string("server_ip", "59.66.143.26", "Server address")
tf.flags.DEFINE_integer("server_port", 8080, "server port")

tf.logging.set_verbosity(tf.logging.INFO)


class InferenceModel(object):
    def __init__(self):
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                       FLAGS.checkpoint_path)
        g.finalize()

        # Create the vocabulary.
        vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
        sess = tf.Session(graph=g)
        restore_fn(sess)
        generator = caption_generator.CaptionGenerator(model, vocab)

        self.vocab = vocab
        self.sess = sess
        self.generator = generator

    def run_inf(self, image_data):
        captions = self.generator.beam_search(self.sess, image_data)
        caption = captions[0]
        sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        return sentence


inf_model = InferenceModel()


class GetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        form_message = """<p>RICGA:please upload a picture(jpeg)</p>
                            <form method="post" action="http://%s:%s" enctype="multipart/form-data">
                            <input name="file" type="file" accept="image/jpeg" />
                            <input name="token" type="hidden" />
                            <input type="submit" value="upload" /></form>""" % (FLAGS.server_ip, FLAGS.server_port)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(form_message.encode('utf-8'))

    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile,
                                headers=self.headers,
                                environ={
                                    'REQUEST_METHOD': 'POST',
                                    'CONTENT_TYPE': self.headers['Content-Type']
                                })
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        for field in form.keys():
            if field == 'file':
                image_file = form[field]
                if image_file.filename:
                    image_data = image_file.file.read()
                    caption = inf_model.run_inf(image_data)
                    # caption = "success"
                    del image_data
                    message = "Caption: %s" % caption
                    self.wfile.write(message.encode("utf-8"))
                    return
        self.wfile.write("failure!!".encode('utf-8'))


def main(_):
    server = HTTPServer(('0.0.0.0', FLAGS.server_port), GetHandler)
    print('Starting server, use <ctrl-c> to stop')
    server.serve_forever()


if __name__ == "__main__":
    tf.app.run()
