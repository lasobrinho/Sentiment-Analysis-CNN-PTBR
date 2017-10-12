#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import KFold, train_test_split
import gensim


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("num_cv_folds", 1, "Number of folds for k-fold cross-validation (default: 4, for holdout set to 1)")
tf.flags.DEFINE_string("positive_data_file", "./Datasets/reaction_cute.data", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./Datasets/reaction_cute_neg.data", "Data source for the negative data.")
tf.flags.DEFINE_string("embeddings_file", "./misc/embeddings/pt/NILC-Embeddings/skip_s300.txt", "Word embeddings file (Gensim/word2vec only).")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Training
# ==================================================
def start_training(x_train, x_dev, 
                   y_train, y_dev, 
                   x_test, y_test, 
                   vocab_processor, 
                   FLAGS, 
                   timestamp, 
                   reaction, 
                   embeddings, 
                   fold=None):

    total_steps = ((len(y_train) // FLAGS.batch_size) + 1) * FLAGS.num_epochs
    
    final_loss, final_accuracy = 0.0, 0.0
    lowest_loss = sys.maxsize

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embeddings=embeddings,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if not fold:
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp)) + "_{:s}".format(reaction)
            else:
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp)) + "_{:s}_fold-{:d}".format(reaction, fold)
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Best model directory
            best_model_dir = os.path.abspath(os.path.join(out_dir, "best_model_dir"))
            best_model_prefix = os.path.join(best_model_dir, "model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                print("{:5d} ({:.2f}%)   loss: {:.6f}   acc: {:8.4f}%".format(step, (step/total_steps) * 100, loss, accuracy * 100))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                print("{:5d} ({:.2f}%)   loss: {:.6f}   acc: {:8.4f}%".format(step, (step/total_steps) * 100, loss, accuracy * 100))
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    loss, accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    # Save the best model found during training (based on the lowest loss)
                    if loss < lowest_loss:
                        lowest_loss = loss
                        path = best_model_saver.save(sess, best_model_prefix, global_step)
                        test_loss, test_accuracy = dev_step(x_test, y_test)
                        with open(os.path.abspath(os.path.join(out_dir, "final_results.txt")), "a") as f:
                            f.write("Step {:d}\n".format(current_step))
                            f.write("  Validation")
                            f.write("    Loss:     {:.6f}\n".format(loss))
                            f.write("    Accuracy: {:8.4f}%\n".format(accuracy * 100))
                            f.write("  Test")
                            f.write("    Loss:     {:.6f}\n".format(test_loss))
                            f.write("    Accuracy: {:8.4f}%\n".format(test_accuracy * 100))
                            f.write("\n")
                        print("\nSaved best model to {}\n".format(path))
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    print("")
            
            print("")
            print("Final Results:")

            print("  Validation")
            final_loss, final_accuracy = dev_step(x_dev, y_dev)
            print("    Loss:     {:.6f}".format(final_loss))
            print("    Accuracy: {:8.4f}%".format(final_accuracy * 100))

            print("  Test: ")
            final_test_loss, final_test_accuracy = dev_step(x_test, y_test)
            print("    Loss:     {:.6f}".format(final_test_loss))
            print("    Accuracy: {:8.4f}%".format(final_test_accuracy * 100))
            print("")

            with open(os.path.abspath(os.path.join(out_dir, "final_results.txt")), "a") as f:
                f.write("\n")
                f.write("Final Results:\n")
                f.write("  Validation\n")
                f.write("    Loss:     {:.6f}\n".format(final_loss))
                f.write("    Accuracy: {:8.4f}%\n".format(final_accuracy * 100))
                f.write("  Test\n")
                f.write("    Loss:     {:.6f}\n".format(final_test_loss))
                f.write("    Accuracy: {:8.4f}%\n".format(final_test_accuracy * 100))
                f.write("\n\n")

    return final_loss, final_accuracy



reactions = ['cute', 'fail', 'hate', 'lol', 'love', 'omg', 'win', 'wtf']
# reactions = ['cute']
for reaction in reactions:

    # Dataset loading
    pos_data_file = "./Datasets/reaction_{:s}.data".format(reaction)
    neg_data_file = "./Datasets/reaction_{:s}_neg.data".format(reaction)

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(pos_data_file, neg_data_file)

    # Build vocabulary
    mean_document_length = np.mean([len(x.split(" ")) for x in x_text])
    stddev_document_length = np.std([len(x.split(" ")) for x in x_text])
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = int(mean_document_length + (2 * stddev_document_length))
    print("Max document length: %d" % max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    emb = []
    embeddings_file = FLAGS.embeddings_file
    if embeddings_file != "":
        print("Loading embeddings file ({:s})...".format(embeddings_file))
        try:
            pt_embeddings = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.embeddings_file, unicode_errors="ignore")
        except:
            print("Error opening embeddings file ({:s})".format(embeddings_file))
            sys.exit()

        ukn_emb = np.random.uniform(-1.0, 1.0, 300)
        ukn_emb = ukn_emb.astype(np.float32)
        for cnt in range(len(vocab_processor.vocabulary_)):
            w = vocab_processor.vocabulary_.reverse(cnt)
            try:
                w_emb = pt_embeddings[w]            
            except:
                w_emb = ukn_emb
            emb.append(w_emb)


    # Randomly shuffle data
    np.random.seed(72854)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

    # input("\nPress any key to start training...")

    n_splits = FLAGS.num_cv_folds
    print()
    print("Starting training on files: ")
    print("Positive: {:s}".format(pos_data_file))
    print("Negative: {:s}".format(neg_data_file))

    all_losses, all_accuracies = [], []
    timestamp = str(int(time.time()))

    if n_splits > 1:
        print("\nUsing {:d}-fold cross-validation".format(n_splits))
        print("Train/Validation for cross-validation: {:d}%/{:d}%\n".format(100 - (100 // n_splits), (100 // n_splits)))
        fold = 1
        kf = KFold(n_splits=n_splits)
        for train_index, val_index in kf.split(x_train):
            print("________________________________________________________________________________")
            print("Starting training for fold {:d}\n".format(fold))
            print()
            cv_x_train, cv_x_val = x_train[train_index], x_train[val_index]
            cv_y_train, cv_y_val = y_train[train_index], y_train[val_index]
            loss, accuracy = start_training(cv_x_train, cv_x_val, 
                                            cv_y_train, cv_y_val, 
                                            x_test, y_test, 
                                            vocab_processor, 
                                            FLAGS, 
                                            timestamp, 
                                            reaction, 
                                            np.asarray(emb), 
                                            fold)
            all_losses.append(loss)
            all_accuracies.append(accuracy)
            fold += 1
    else:
        print("\nUsing holdout validation (train/val/test: 60/20/20)")
        print("________________________________________________________________________________\n")
        hold_x_train, hold_x_val, hold_y_train, hold_y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=False)
        loss, accuracy = start_training(hold_x_train, hold_x_val, 
                                        hold_y_train, hold_y_val, 
                                        x_test, y_test, 
                                        vocab_processor, 
                                        FLAGS, 
                                        timestamp, 
                                        reaction, 
                                        np.asarray(emb))
        all_losses.append(loss)
        all_accuracies.append(accuracy)


    print("________________________________________________________________________________")
    print("Final results:")
    print("Average loss: {:.6f}".format(np.mean(all_losses)))
    print("Average accuracy: {:8.4f}%".format(np.mean(all_accuracies) * 100))

    print()
    print("Finished training operation")
    print("================================================================================")
