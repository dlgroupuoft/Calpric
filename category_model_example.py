"""
customizable example for the active learning & crowdsourcing component: a Calpric category model
"""

import numpy as np
from os.path import join
import pandas as pd
import math

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
import pickle


########################################### settings ###########################################

strategies = uncertainty_batch_sampling
CHAR_SET = 'utf-8-sig'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


########################################### checking algorithm ###########################################

def threshold_filter(predictions, y_true):
    thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for val in thresholds:
        y_pred=predictions.copy()

        y_pred[y_pred>=val]=1
        y_pred[y_pred<val]=0

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        f1 = f1(y_true, y_pred)

        print("Micro-average quality numbers")
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))




########################################### Evaluation and metrics ###########################################

def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())


def accuracy(y_true, y_pred):
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)))
    return accuracy


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2 * ((precision_ * recall_) / (precision_ + recall_ + K.epsilon()))





########################################### active learning ###########################################

def active_learning_cli(p_id, category, BOOTSTRAP_ALIGNMENT_RATE, THRESHOLD=0.8, NUM_TURK=5, MAX_WORDS=400,
                        TOTAL_ITERATIONS=500, QUERIES_PER_ITERATION=42, EPOCHS=48, STOPPING_CRITERIA=500,
                        __TEST__=True):

    print("################### active_learning_cli() ###################")
    print("initializing active_learning_cli()...")


    '''
        prepare test sets: natural and balanced
    '''
    test_sent_label_tuple_list = []
    test_balance_sent_label_tuple_list = []

    '''
    prepare the initial train set to bootstrap
    '''
    ini_sent_label_tuple_list = []

    '''
    training pool pre-processing: prepare the unlabeled training pool
    '''
    pool_sent_list = []

    '''
    Create embedding matrix.
    embedding_bank: pre-trained embedding
    sents_list: sanitized sentences list
    sents_sequences: padded word index list
    embedding_matrix: word index to embedding vector matrix
    '''
    t = Tokenizer(oov_token='UNK')

    '''create your bert-based model'''
    bert_model = ''
    model = KerasClassifier(bert_model)

    '''
    convert to sentence vectors
    '''
    init_sent_list = [pair[0] for pair in ini_sent_label_tuple_list]
    y_init_train = [pair[1] for pair in ini_sent_label_tuple_list]
    X_init_train = pad_sequences(t.texts_to_sequences(init_sent_list), maxlen=MAX_WORDS, truncating='post',
                                 padding='post')

    test_sent_list = [pair[0] for pair in test_sent_label_tuple_list]
    y_test = [pair[1] for pair in test_sent_label_tuple_list]
    X_test = pad_sequences(t.texts_to_sequences(test_sent_list), maxlen=MAX_WORDS, truncating='post', padding='post')

    test_balance_sent_list = [pair[0] for pair in test_balance_sent_label_tuple_list]
    y_test_balance = [pair[1] for pair in test_balance_sent_label_tuple_list]
    X_test_balance = pad_sequences(t.texts_to_sequences(test_balance_sent_list), maxlen=MAX_WORDS, truncating='post', padding='post')

    pool_sent_sequences = pad_sequences(t.texts_to_sequences(pool_sent_list), maxlen=MAX_WORDS, truncating='post',
                                        padding='post')

    # logs for training size and performance evaluation
    current_train_size = len(y_init_train)
    current_query_size = len(y_init_train) / BOOTSTRAP_ALIGNMENT_RATE  # get from mTurk results
    num_pos = np.count_nonzero(y_init_train)

    '''
       Active Learning: initialization
    '''

    learner = ActiveLearner(
        estimator=model,  # need to wrap the classifier depending on the model type you select
        query_strategy=strategies,
        X_training=X_init_train, y_training=y_init_train,
        verbose=1,
    )

    test_pos = np.count_nonzero(y_test)
    test_neg_percent = 1. - test_pos / float(len(y_test))
    print("The number of positive samples in test set: " + str(test_pos))
    print("The percentage of negative sample is: " + str(test_neg_percent))

    test_evaluation = learner.score(X_test, y_test, verbose=1)
    predictions = learner.predict(X_test)
    threshold_filter(predictions, y_test)

    '''
    Active Learning: active querying
    '''
    YOUR_PATH = ''
    MODEL_SAVE_DIR = join(YOUR_PATH, 'saved_data/classifier_models/')
    TOKENIZER_SAVE_DIR = join(YOUR_PATH, 'saved_data/tokenizer/')
    X_POOL_SAVE_DIR = join(YOUR_PATH, 'saved_data/X_pool/')
    X_POOL_SENTENCES_DIR = join(YOUR_PATH, 'saved_data/X_pool_sentences/')

    # active learning loop
    X_pool = pool_sent_sequences
    for idx in range(TOTAL_ITERATIONS):
        # save model
        MODEL_FILE = category + '_model_' + str(idx)
        MODEL_SAVE_PATH = join(MODEL_SAVE_DIR, MODEL_FILE)
        model.model.save(MODEL_SAVE_PATH)

        # save X_pool and sentences
        X_POOL_FILE = category + str(idx) + '_X_pool.pickle'
        X_POOL_SAVE_PATH = join(X_POOL_SAVE_DIR, X_POOL_FILE)
        with open(X_POOL_SAVE_PATH, 'wb') as handle:
            pickle.dump(X_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)

        save_sentences = pool_sent_list

        sentences_FILE = category + str(idx) + '_sentences.pickle'
        sentences_SAVE_PATH = join(X_POOL_SENTENCES_DIR, sentences_FILE)
        with open(sentences_SAVE_PATH, 'wb') as handle:
            pickle.dump(save_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save tokenizer
        TOKENIZER_FILE = category + str(idx) + '_tokenizer.pickle'
        TOKENIZER_SAVE_PATH = join(TOKENIZER_SAVE_DIR, TOKENIZER_FILE)
        with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # can stop the training for each [preset#] of iterations
        if idx % 100 == 0 and idx != 0:
            print("the current number of iterations (starts with 0) is: " + str(idx))
            still_go = input(
                "Do you want to stop the training process? If yes, type (yes please stop), else type (n).")
            if still_go == 'yes please stop':
                break


        ##################################################### reloading part ends #############################################

        if X_pool.size < STOPPING_CRITERIA:
            raise Exception("less than {0} segments in the unlabelled pool".format(STOPPING_CRITERIA))

        '''base query'''
        queryID = idx + 1
        print('Query no. %d' % queryID)
        print("before querying: check len difference, X_pool vs. pool_sent_list")
        print(len(X_pool))
        print(len(pool_sent_list))

        base_query_idx, base_query_instance = learner.query(X_pool, n_instances=QUERIES_PER_ITERATION)
        print("base_query_instance: ")
        print(base_query_instance)

        '''get the text back and post it to mTurk'''
        selected_texts_list = [pool_sent_list[i] for i in base_query_idx.tolist()]

        pre_selected_texts_set = set(selected_texts_list)
        post_selected_texts_set = set()
        for each_text in pre_selected_texts_set:
            if not check_english(str(each_text)):
                print("found one with non english: " + str(each_text))
            else:
                post_selected_texts_set.add(each_text)

        X_pool = np.delete(X_pool, base_query_idx, axis=0)
        pool_sent_list = [value for i, value in enumerate(pool_sent_list) if i not in base_query_idx.tolist()]

        '''2nd (next) query'''
        while True:
            if len(post_selected_texts_set) >= QUERIES_PER_ITERATION:
                print("current number of selected texts: " + str(len(post_selected_texts_set)))
                break
            next_query_idx, next_query_instance = \
                learner.query(X_pool, n_instances=(QUERIES_PER_ITERATION - len(post_selected_texts_set)))
            next_query_texts = [pool_sent_list[i] for i in next_query_idx.tolist()]
            next_pre_selected_texts_set = set(next_query_texts)

            X_pool = np.delete(X_pool, next_query_idx, axis=0)
            pool_sent_list = [value for i, value in enumerate(pool_sent_list) if i not in next_query_idx.tolist()]

            '''3rd (last) query'''
            for each_text in next_pre_selected_texts_set:
                single_text = each_text
                copy_post_selected_texts_set = post_selected_texts_set
                while (not check_english(str(single_text))) or (str(single_text) in copy_post_selected_texts_set):
                    if single_text[0] in post_selected_texts_set:
                        break
                    single_idx, single_instance = learner.query(X_pool, n_instances=1)

                    # only 1 instance obtained here, so just [0]
                    single_text_list = [pool_sent_list[i] for i in single_idx.tolist()]
                    single_text = single_text_list[0]
                    X_pool = np.delete(X_pool, single_idx, axis=0)
                    pool_sent_list = [value for i, value in enumerate(pool_sent_list) if
                                      i not in single_idx.tolist()]

                post_selected_texts_set.add(single_text)
                print("length of post_selected_texts_set (end of for loop): " + str(len(post_selected_texts_set)))


        # ============================================ obtain new labels ==============================================
        # ...asking for new labels from the Oracle... supply label for queried instance
        pass_read_label_test = False
        pass_input_test = False
        df_new = None
        while not pass_read_label_test or not pass_input_test or df_new is None:
            raw_result_path = input("Please enter the path to mturk results (to be consolidated):")
            print("Your entered path: " + raw_result_path)
            try:
                # consolidate results (from a file)
                pass_input_test, label_path = process_42policies_simple_results. \
                    process_42_simple_mturk_results(raw_result_path, CATEGORY=category, query_id=queryID,
                                                    POLICY_PER_HIT=QUERIES_PER_ITERATION,
                                                    NUM_TURK=NUM_TURK, AT_THRESHOLD=THRESHOLD)
            except Exception as process_err:
                print("Error: input raw result file processing error!")
                print(process_err)

            # from label_path get policies & labels
            if pass_input_test:
                try:
                    df_new = pd.read_csv(label_path, encoding=CHAR_SET)
                except Exception as new_input_err:
                    print("Error: consolidated label file read error!")
                    print(new_input_err)

        # write your own function to collect the processed labels from crowdsourcers
        processed_labels = []
        print("processed_labels: " + str(processed_labels))

        new_add_X = []
        new_add_y = []
        for each_label in processed_labels:
            new_add_X.append(each_label['X_segment_text'])
            new_add_y.append(each_label['Y_label'])

        new_add_sent_sequences = pad_sequences(t.texts_to_sequences(new_add_X), maxlen=MAX_WORDS, truncating='post',
                                               padding='post')

        # ========================================== done obtain new labels ===========================================

        current_query_size += QUERIES_PER_ITERATION
        current_train_size += len(new_add_y)

        # none consolidated
        if len(new_add_y) == 0:
            AL_neg_percent = 1. - float(num_pos) / float(current_train_size)
            line_chart_list = [current_query_size, current_train_size, AL_neg_percent]

            test_evaluation = learner.score(X_test, y_test, verbose=1)
            print("warning: the newly added batch is empty")
            print("evaluation: loss, accuracy, precision, recall, f1, mcor")
            print(test_evaluation)

            test_evaluation_balance = learner.score(X_test_balance, y_test_balance, verbose=1)
            print("evaluation_balance: loss, accuracy, precision, recall, f1, mcor")
            print(test_evaluation_balance)
            continue

        new_add_y = np.array(new_add_y)
        num_pos += np.count_nonzero(new_add_y)

        X_selected = new_add_sent_sequences
        y_selected = new_add_y

        learner.teach(
            X=X_selected, y=y_selected, epochs=EPOCHS, only_new=True,
            verbose=1
        )

        # test current model
        test_evaluation = learner.score(X_test, y_test, verbose=1)
        test_evaluation_balance = learner.score(X_test_balance, y_test_balance, verbose=1)

        if isinstance(test_evaluation, list):
            if math.isnan(test_evaluation[0]):
                break
        else:
            if math.isnan(test_evaluation):
                break


    print("---------------- Final Evaluation ----------------")
    test_evaluation = learner.score(X_test, y_test, verbose=1)
    test_evaluation_balance = learner.score(X_test_balance, y_test_balance, verbose=1)

    print("completing active_learning_cli()...")
    print("######################## end active_learning_cli() ########################")
