#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys
import gensim

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("runs_dir", "./runs", "Directory with trained models")
tf.flags.DEFINE_string("input_text", "Os brasileiros estão amando ver a Chloe coberta de Paçoquita no Instagram.  E ela fez aquela inconfundível cara de Chloe está te julgando..  1. A Chloe menininha americana que ficou famosa na internet pela sua cara de ATA recebeu um brindezão da Paçoquita. 2. E a caixa de comentários foi tomada em uma só voz pelo internauta brasileiro. 3. Se você acha que já viu essa carinha antes é porque provavelmente viu mesmo neste meme. A imagem vem deste vídeo no qual a menininha não acha lá essas coisas o fato de ir para a Disney. 4. Ela também aparece neste vídeo no qual fica repetindo várias vezes “PACOQUITA” que é como ela pronuncia o nome do doce. 5. No ano passado no Instagram da Chloe já havia sido publicado um vídeo dela tentando falar essa palavra em português. Nesse até que ela se sai bem. 6. O histórico com quitutes brasileiros não acaba aí: aqui está a Lily irmãzinha da Chloe curtindo um “Brazilian cheese bread” também conhecido como pão de queijo.", "Input text to be classified")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("embeddings_file", "./misc/embeddings/pt/NILC-Embeddings/skip_s300.txt", "Word embeddings file (Gensim/word2vec only).")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value).encode('utf-8').decode(sys.stdout.encoding))
print("")

x_raw = [data_helpers.clean_str(FLAGS.input_text)]

emb = []
embeddings_file = FLAGS.embeddings_file
if embeddings_file != "":
    print("Loading embeddings file... ({:s})".format(embeddings_file))
    try:
        pt_embeddings = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.embeddings_file, unicode_errors="ignore")
    except:
        print("Error opening embeddings file ({:s})".format(embeddings_file))
        sys.exit()



print("\nPredicting...\n")

# Prediction
# ==================================================

models_dir = os.listdir(FLAGS.runs_dir)
reaction_probabilities = {}

for model_dir in models_dir:
    reaction_name = model_dir.split('_')[1]
    checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.runs_dir, model_dir, 'best_model_dir'))

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.runs_dir, model_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    
    if embeddings_file != "":
        ukn_emb = np.random.uniform(-1.0, 1.0, 300)
        ukn_emb = ukn_emb.astype(np.float32)
        for cnt in range(len(vocab_processor.vocabulary_)):
            w = vocab_processor.vocabulary_.reverse(cnt)
            try:
                w_emb = pt_embeddings[w]            
            except:
                w_emb = ukn_emb
            emb.append(w_emb)


    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            #predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            final_scores = sess.run(scores, {input_x: x_test, dropout_keep_prob: 1.0})

    reaction_probabilities[reaction_name] = final_scores

final_reaction = ('', -100)
for reaction, probabilities in reaction_probabilities.items():
    pos_probability = probabilities[0, 1]
    if pos_probability > final_reaction[1]:
        final_reaction = (reaction, pos_probability)

print("")
print("Reaction: {:s}".format(final_reaction[0]))
print("")
