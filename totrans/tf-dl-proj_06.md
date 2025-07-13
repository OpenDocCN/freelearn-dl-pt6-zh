
# Create and Train Machine Translation Systems

The objective of this project is to train an **artificial intelligence** (**AI**) model to be able to translate between two languages. Specifically, we will see an automatic translator which reads German and produces English sentences; although, the model and the code developed in this chapter is generic enough for any language pair.

The project explored in this chapter has four important sections, as follows:

*   A walkthrough of the architecture
*   Preprocessing the corpora
*   Training the machine translator
*   Testing and translating

Each of them will describe one key component of the project, and, at the end, you'll have a clear picture of what's going on.

# A walkthrough of the architecture

A machine translation system receives as input an arbitrary string in one language and produces, as output, a string with the same meaning but in another language. Google Translate is one example (but also many other main IT companies have their own). There, users are able to translate to and from more than 100 languages. Using the webpage is easy: on the left just put the sentence you want to translate (for example, Hello World), select its language (in the example, it's English), and select the language you want it to be translated to.

Here's an example where we translate the sentence Hello World to French:

![](img/ccea21d1-123d-41a1-b60d-7cae15f0e6cf.png)

Is it easy? At a glance, we may think it's a simple dictionary substitution. Words are chunked, the translation is looked up on the specific English-to-French dictionary, and each word is substituted with its translation. Unfortunately, that's not the case. In the example, the English sentence has two words, while the French one has three. More generically, think about phrasal verbs (turn up, turn off, turn on, turn down), Saxon genitive, grammatical gender, tenses, conditional sentences... they don't always have a direct translation, and the correct one should follow the context of the sentence.

That's why, for doing machine translation, we need some artificial intelligence tools. Specifically, as for many other **natural language processing** (**NLP**) tasks, we'll be using **recurrent neural networks** (**RNNs**). We introduced RNNs in the previous chapter, and the main feature they have is that they work on sequences: given an input sequence, they produce an output sequence. The objective of this chapter is to create the correct training pipeline for having a sentence as the input sequence, and its translation as the output one. Remember also the *no free lunch theorem*: this process isn't easy, and more solutions can be created with the same result. Here, in this chapter, we will propose a simple but powerful one.

First of all, we start with the corpora: it's maybe the hardest thing to find, since it should contain a high fidelity translation of many sentences from a language to another one. Fortunately, NLTK, a well-known package of Python for NLP, contains the corpora Comtrans. **Comtrans** is the acronym of **combination approach to machine translation**, and contains an aligned corpora for three languages: German, French, and English.

In this project, we will use these corpora for a few reasons, as follows:

1.  It's easy to download and import in Python.
2.  No preprocessing is needed to read it from disk / from the internet. NLTK already handles that part.
3.  It's small enough to be used on many laptops (a few dozen thousands sentences).
4.  It's freely available on the internet.

For more information about the Comtrans project, go to [http://www.fask.uni-mainz.de/user/rapp/comtrans/](http://www.fask.uni-mainz.de/user/rapp/comtrans/).

More specifically, we will try to create a machine translation system to translate German to English. We picked these two languages at random among the ones available in the Comtrans corpora: feel free to flip them, or use the French corpora instead. The pipeline of our project is generic enough to handle any combination.

Let's now investigate how the corpora is organized by typing some commands:

```py
from nltk.corpus import comtrans
print(comtrans.aligned_sents('alignment-de-en.txt')[0])
```

The output is as follows:

```py
<AlignedSent: 'Wiederaufnahme der S...' -> 'Resumption of the se...'>
```

The pairs of sentences are available using the function `aligned_sents`. The filename contains the from and to language. In this case, as for the following part of the project, we will translate German (*de*) to English (*en*). The returned object is an instance of the class `nltk.translate.api.AlignedSent`. By looking at the documentation, the first language is accessible with the attribute `words`, while the second language is accessible with the attribute `mots`. So, to extract the German sentence and its English translation separately, we should run:

```py
print(comtrans.aligned_sents()[0].words)
print(comtrans.aligned_sents()[0].mots)
```

The preceding code outputs:

```py
['Wiederaufnahme', 'der', 'Sitzungsperiode']
['Resumption', 'of', 'the', 'session']
```

How nice! The sentences are already tokenized, and they look as sequences. In fact, they will be the input and (hopefully) the output of the RNN which will provide the service of machine translation from German to English for our project.

Furthermore, if you want to understand the dynamics of the language, Comtrans makes available the alignment of the words in the translation:

```py
print(comtrans.aligned_sents()[0].alignment)
```

The preceding code outputs:

```py
0-0 1-1 1-2 2-3
```

The first word in German is translated to the first word in English *(Wiederaufnahme* to *Resumption),* the second to the second *(der* to both *of* and *the),* and the third (at index 1) is translated with the fourth *(Sitzungsperiode* to *session).*

# Preprocessing of the corpora

The first step is to retrieve the corpora. We've already seen how to do this, but let's now formalize it in a function. To make it generic enough, let's enclose these functions in a file named `corpora_tools.py`.

1.  Let's do some imports that we will use later on:

```py
import pickle
import re
from collections import Counter
from nltk.corpus import comtrans
```

2.  Now, let's create the function to retrieve the corpora:

```py
def retrieve_corpora(translated_sentences_l1_l2='alignment-de-en.txt'):
    print("Retrieving corpora: {}".format(translated_sentences_l1_l2))
    als = comtrans.aligned_sents(translated_sentences_l1_l2)
    sentences_l1 = [sent.words for sent in als]
    sentences_l2 = [sent.mots for sent in als]
    return sentences_l1, sentences_l2
```

This function has one argument; the file containing the aligned sentences from the NLTK Comtrans corpora. It returns two lists of sentences (actually, they're a list of tokens), one for the source language (in our case, German), the other in the destination language (in our case, English).

3.  On a separate Python REPL, we can test this function:

```py
sen_l1, sen_l2 = retrieve_corpora()
print("# A sentence in the two languages DE & EN")
print("DE:", sen_l1[0])
print("EN:", sen_l2[0])
print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)
```

4.  The preceding code creates the following output:

```py
Retrieving corpora: alignment-de-en.txt
# A sentence in the two languages DE & EN
DE: ['Wiederaufnahme', 'der', 'Sitzungsperiode']
EN: ['Resumption', 'of', 'the', 'session']
# Corpora length (i.e. number of sentences)
33334
```

We also printed the number of sentences in each corpora (33,000) and asserted that the number of sentences in the source and the destination languages is the same.

5.  In the following step, we want to clean up the tokens. Specifically, we want to tokenize punctuation and lowercase the tokens. To do so, we can create a new function in `corpora_tools.py`. We will use the `regex` module to perform the further splitting tokenization:

```py
def clean_sentence(sentence):
    regex_splitter = re.compile("([!?.,:;$\"')( ])")
    clean_words = [re.split(regex_splitter, word.lower()) for word in sentence]
    return [w for words in clean_words for w in words if words if w]
```

6.  Again, in the REPL, let's test the function:

```py
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
print("# Same sentence as before, but chunked and cleaned")
print("DE:", clean_sen_l1[0])
print("EN:", clean_sen_l2[0])
```

The preceding code outputs the same sentence as before, but chunked and cleaned:

```py
DE: ['wiederaufnahme', 'der', 'sitzungsperiode']
EN: ['resumption', 'of', 'the', 'session']
```

Nice!

The next step for this project is filtering the sentences that are too long to be processed. Since our goal is to perform the processing on a local machine, we should limit ourselves to sentences up to *N* tokens. In this case, we set *N*=20, in order to be able to train the learner within 24 hours. If you have a powerful machine, feel free to increase that limit. To make the function generic enough, there's also a lower bound with a default value set to 0, such as an empty token set.

1.  The logic of the function is very easy: if the number of tokens for a sentence or its translation is greater than *N*, then the sentence (in both languages) is removed:

```py
def filter_sentence_length(sentences_l1, sentences_l2, min_len=0, max_len=20):
    filtered_sentences_l1 = []
    filtered_sentences_l2 = []
    for i in range(len(sentences_l1)):
        if min_len <= len(sentences_l1[i]) <= max_len and \
                 min_len <= len(sentences_l2[i]) <= max_len:
            filtered_sentences_l1.append(sentences_l1[i])
            filtered_sentences_l2.append(sentences_l2[i])
    return filtered_sentences_l1, filtered_sentences_l2
```

2.  Again, let's see in the REPL how many sentences survived this filter. Remember, we started with more than 33,000:

```py
filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, 
          clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```

The preceding code prints the following output:

```py
# Filtered Corpora length (i.e. number of sentences)
14788
```

Almost 15,000 sentences survived, that is, half of the corpora.

Now, we finally move from text to numbers (which AI mainly uses). To do so, we shall create a dictionary of the words for each language. The dictionary should be big enough to contain most of the words, though we can discard some if the language has words with low occourrence. This is a common practice even in the tf-idf (term frequency within a document, multiplied by the inverse of the document frequency, i.e. in how many documents that token appears), where very rare words are discarded to speed up the computation, and make the solution more scalable and generic. We need here four special symbols in both dictionaries:

1.  One symbol for padding (we'll see later why we need it)
2.  One symbol for dividing the two sentences
3.  One symbol to indicate where the sentence stops
4.  One symbol to indicate unknown words (like the very rare ones)

For doing so, let's create a new file named `data_utils.py` containing the following lines of code:

```py
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]
```

Then, back to the `corpora_tools.py` file, let's add the following function:

```py
import data_utils

def create_indexed_dictionary(sentences, dict_size=10000, storage_path=None):
    count_words = Counter()
    dict_words = {}
    opt_dict_size = len(data_utils.OP_DICT_IDS)
    for sen in sentences:
        for word in sen:
            count_words[word] += 1

    dict_words[data_utils._PAD] = data_utils.PAD_ID
    dict_words[data_utils._GO] = data_utils.GO_ID
    dict_words[data_utils._EOS] = data_utils.EOS_ID
    dict_words[data_utils._UNK] = data_utils.UNK_ID

    for idx, item in enumerate(count_words.most_common(dict_size)):
        dict_words[item[0]] = idx + opt_dict_size
    if storage_path:
        pickle.dump(dict_words, open(storage_path, "wb"))
    return dict_words
```

This function takes as arguments the number of entries in the dictionary and the path of where to store the dictionary. Remember, the dictionary is created while training the algorithms: during the testing phase it's loaded, and the association token/symbol should be the same one as used in the training. If the number of unique tokens is greater than the value set, only the most popular ones are selected. At the end, the dictionary contains the association between a token and its ID for each language.

After building the dictionary, we should look up the tokens and substitute them with their token ID.

For that, we need another function:

```py
def sentences_to_indexes(sentences, indexed_dictionary):
    indexed_sentences = []
    not_found_counter = 0
    for sent in sentences:
        idx_sent = []
        for word in sent:
            try:
                idx_sent.append(indexed_dictionary[word])
            except KeyError:
                idx_sent.append(data_utils.UNK_ID)
                not_found_counter += 1
        indexed_sentences.append(idx_sent)

    print('[sentences_to_indexes] Did not find {} words'.format(not_found_counter))
    return indexed_sentences
```

This step is very simple; the token is substituted with its ID. If the token is not in the dictionary, the ID of the unknown token is used. Let's see in the REPL how our sentences look after these steps:

```py
dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path="/tmp/l2_dict.p")
idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
print("# Same sentences as before, with their dictionary ID")
print("DE:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
```

This code prints the token and its ID for both the sentences. What's used in the RNN will be just the second element of each tuple, that is, the integer ID:

```py
# Same sentences as before, with their dictionary ID
DE: [('wiederaufnahme', 1616), ('der', 7), ('sitzungsperiode', 618)]
EN: [('resumption', 1779), ('of', 8), ('the', 5), ('session', 549)]
```

Please also note how frequent tokens, such as *the* and *of* in English, and *der* in German, have a low ID. That's because the IDs are sorted by popularity (see the body of the function `create_indexed_dictionary`).

Even though we did the filtering to limit the maximum size of the sentences, we should create a function to extract the maximum size. For the lucky owners of very powerful machines, which didn't do any filtering, that's the moment to see how long the longest sentence in the RNN will be. That's simply the function:

```py
def extract_max_length(corpora):
    return max([len(sentence) for sentence in corpora])
```

Let's apply the following to our sentences:

```py
max_length_l1 = extract_max_length(idx_sentences_l1)
max_length_l2 = extract_max_length(idx_sentences_l2)
print("# Max sentence sizes:")
print("DE:", max_length_l1)
print("EN:", max_length_l2)
```

As expected, the output is:

```py
# Max sentence sizes:
DE: 20
EN: 20
```

The final preprocessing step is padding. We need all the sequences to be the same length, therefore we should pad the shorter ones. Also, we need to insert the correct tokens to instruct the RNN where the string begins and ends.

Basically, this step should:

*   Pad the input sequences, for all being 20 symbols long
*   Pad the output sequence, to be 20 symbols long
*   Insert an `_GO` at the beginning of the output sequence and an `_EOS` at the end to position the start and the end of the translation

This is done by this function (insert it in the `corpora_tools.py`):

```py
def prepare_sentences(sentences_l1, sentences_l2, len_l1, len_l2):
    assert len(sentences_l1) == len(sentences_l2)
    data_set = []
    for i in range(len(sentences_l1)):
        padding_l1 = len_l1 - len(sentences_l1[i])
        pad_sentence_l1 = ([data_utils.PAD_ID]*padding_l1) + sentences_l1[i]
        padding_l2 = len_l2 - len(sentences_l2[i])
        pad_sentence_l2 = [data_utils.GO_ID] + sentences_l2[i] + [data_utils.EOS_ID] + ([data_utils.PAD_ID] * padding_l2)
        data_set.append([pad_sentence_l1, pad_sentence_l2])
    return data_set
```

To test it, let's prepare the dataset and print the first sentence:

```py
data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
print("# Prepared minibatch with paddings and extra stuff")
print("DE:", data_set[0][0])
print("EN:", data_set[0][1])
print("# The sentence pass from X to Y tokens")
print("DE:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("EN:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
```

The preceding code outputs the following:

```py
# Prepared minibatch with paddings and extra stuff
DE: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7, 618]
EN: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# The sentence pass from X to Y tokens
DE: 3 -> 20
EN: 4 -> 22
```

As you can see, the input and the output are padded with zeros to have a constant length (in the dictionary, they correspond to `_PAD`, see `data_utils.py`), and the output contains the markers 1 and 2 just before the start and the end of the sentence. As proven effective in the literature, we're going to pad the input sentences at the start and the output sentences at the end. After this operation, all the input sentences are `20` items long, and the output sentences `22`.

# Training the machine translator

So far, we've seen the steps to preprocess the corpora, but not the model used. The model is actually already available on the TensorFlow Models repository, freely downloadable from [https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py.).

The piece of code is licensed with Apache 2.0\. We really thank the authors for having open sourced such a great model. Copyright 2015 The TensorFlow Authors. All Rights Reserved. Licensed under the Apache License, Version 2.0 (the License); You may not use this file except in compliance with the License. You may obtain a copy of the License at: [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) Unless required by applicable law or agreed to in writing, software. Distributed under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

We will see the usage of the model throughout this section. First, let's create a new file named `train_translator.py` and put in some imports and some constants. We will save the dictionary in the `/tmp/` directory, as well as the model and its checkpoints:

```py
import time
import math
import sys
import pickle
import glob
import os
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from corpora_tools import *

path_l1_dict = "/tmp/l1_dict.p"
path_l2_dict = "/tmp/l2_dict.p"
model_dir = "/tmp/translate "
model_checkpoints = model_dir + "/translate.ckpt"
```

Now, let's use all the tools created in the previous section within a function that, given a Boolean flag, returns the corpora. More specifically, if the argument is `False`, it builds the dictionary from scratch (and saves it); otherwise, it uses the dictionary available in the path:

```py
def build_dataset(use_stored_dictionary=False):
    sen_l1, sen_l2 = retrieve_corpora()
    clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
    clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
    filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2)

    if not use_stored_dictionary:
        dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path=path_l1_dict)
        dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path=path_l2_dict)
    else:
        dict_l1 = pickle.load(open(path_l1_dict, "rb"))
        dict_l2 = pickle.load(open(path_l2_dict, "rb"))

    dict_l1_length = len(dict_l1)
    dict_l2_length = len(dict_l2)

    idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
    idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)

    max_length_l1 = extract_max_length(idx_sentences_l1)
    max_length_l2 = extract_max_length(idx_sentences_l2)

    data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
    return (filt_clean_sen_l1, filt_clean_sen_l2), \
        data_set, \
        (max_length_l1, max_length_l2), \
        (dict_l1_length, dict_l2_length)
```

This function returns the cleaned sentences, the dataset, the maximum length of the sentences, and the lengths of the dictionaries.

Also, we need to have a function to clean up the model. Every time we run the training routine we need to clean up the model directory, as we haven't provided any garbage information. We can do this with a very simple function:

```py
def cleanup_checkpoints(model_dir, model_checkpoints):
    for f in glob.glob(model_checkpoints + "*"):
    os.remove(f)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
```

Finally, let's create the model in a reusable fashion:

```py
def get_seq2seq_model(session, forward_only, dict_lengths, max_sentence_lengths, model_dir):
    model = Seq2SeqModel(
            source_vocab_size=dict_lengths[0],
            target_vocab_size=dict_lengths[1],
            buckets=[max_sentence_lengths],
            size=256,
            num_layers=2,
            max_gradient_norm=5.0,
            batch_size=64,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            forward_only=forward_only,
            dtype=tf.float16)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model
```

This function calls the constructor of the model, passing the following parameters:

*   The source vocabulary size (German, in our example)
*   The target vocabulary size (English, in our example)
*   The buckets (in our example is just one, since we padded all the sequences to a single size)
*   The **long short-term memory** (**LSTM**) internal units size
*   The number of stacked LSTM layers
*   The maximum norm of the gradient (for gradient clipping)
*   The mini-batch size (that is, how many observations for each training step)
*   The learning rate
*   The learning rate decay factor
*   The direction of the model
*   The type of data (in our example, we will use flat16, that is, float using 2 bytes)

To make the training faster and obtain a model with good performance, we have already set the values in the code; feel free to change them and see how it performs.

The final if/else in the function retrieves the model, from its checkpoint, if the model already exists. In fact, this function will be used in the decoder too to retrieve and model on the test set.

Finally, we have reached the function to train the machine translator. Here it is:

```py
def train():
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, False, dict_lengths, max_sentence_lengths, model_dir)
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        bucket = 0
        steps_per_checkpoint = 100
        max_steps = 20000
        while current_step < max_steps:
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step {} learning rate {} step-time {} perplexity {}".format(
                model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                sess.run(model.learning_rate_decay_op)
                model.saver.save(sess, model_checkpoints, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                print(" eval: perplexity {}".format(eval_ppx))
                sys.stdout.flush()    
```

The function starts by creating the model. Also, it sets some constants on the steps per checkpoints and the maximum number of steps. Specifically, in the code, we will save a model every 100 steps and we will perform no more than 20,000 steps. If it still takes too long, feel free to kill the program: every checkpoint contains a trained model, and the decoder will use the most updated one.

At this point, we enter the while loop. For each step, we ask the model to get a minibatch of data (of size 64, as set previously). The method `get_batch` returns the inputs (that is, the source sequence), the outputs (that is, the destination sequence), and the weights of the model. With the method `step`, we run one step of the training. One piece of information returned is the loss for the current minibatch of data. That's all the training!

To report the performance and store the model every 100 steps, we print the average perplexity of the model (the lower, the better) on the 100 previous steps, and we save the checkpoint. The perplexity is a metric connected to the uncertainty of the predictions: the more confident we're about the tokens, the lower will be the perplexity of the output sentence. Also, we reset the counters and we extract the same metric from a single minibatch of the test set (in this case, it's a random minibatch of the dataset), and performances of it are printed too. Then, the training process restarts again.

As an improvement, every 100 steps we also reduce the learning rate by a factor. In this case, we multiply it by 0.99\. This helps the convergence and the stability of the training.

We now have to connect all the functions together. In order to create a script that can be called by the command line but is also used by other scripts to import functions, we can create a `main`, as follows:

```py
if __name__ == "__main__":
    _, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    cleanup_checkpoints(model_dir, model_checkpoints)
    train()
```

In the console, you can now train your machine translator system with a very simple command:

```py
$> python train_translator.py
```

On an average laptop, without an NVIDIA GPU, it takes more than a day to reach a perplexity below 10 (12+ hours). This is the output:

```py
Retrieving corpora: alignment-de-en.txt
[sentences_to_indexes] Did not find 1097 words
[sentences_to_indexes] Did not find 0 words
Created model with fresh parameters.
global step 100 learning rate 0.5 step-time 4.3573073434829713 perplexity 526.6638556683066
eval: perplexity 159.2240770935855
[...]
global step 10500 learning rate 0.180419921875 step-time 4.35106209993362414 perplexity 2.0458043055629487
eval: perplexity 1.8646006006241982
[...]
```

# Test and translate

The code for the translation is in the file `test_translator.py`.

We start with some imports and the location of the pre-trained model:

```py
import pickle
import sys
import numpy as np
import tensorflow as tf
import data_utils
from train_translator import (get_seq2seq_model, path_l1_dict, path_l2_dict,
build_dataset)
model_dir = "/tmp/translate"
```

Now, let's create a function to decode the output sequence generated by the RNN. Mind that the sequence is multidimensional, and each dimension corresponds to the probability of that word, therefore we will pick the most likely one. With the help of the reverse dictionary, we can then figure out what was the actual word. Finally, we will trim the markings (padding, start, end of string) and print the output.

In this example, we will decode the first five sentences in the training set, starting from the raw corpora. Feel free to insert new strings or use different corpora:

```py
def decode():
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
        model.batch_size = 1
        bucket = 0
        for idx in range(len(data_set))[:5]:
            print("-------------------")
            print("Source sentence: ", sentences[0][idx])
            print("Source tokens: ", data_set[idx][0])
            print("Ideal tokens out: ", data_set[idx][1])
            print("Ideal sentence out: ", sentences[1][idx])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            {bucket: [(data_set[idx][0], [])]}, bucket)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
            target_weights, bucket, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if data_utils.EOS_ID in outputs:
                outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
            print("Model output: ", " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs]))
            sys.stdout.flush()
```

Here, again, we need a `main` to work with the command line, as follows:

```py
if __name__ == "__main__":
    dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    inv_dict_l2 = {v: k for k, v in dict_l2.items()}
    build_dataset(True)
    sentences, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    try:
        print("Reading from", model_dir)
        print("Dictionary lengths", dict_lengths)
        print("Bucket size", max_sentence_lengths)
    except NameError:
        print("One or more variables not in scope. Translation not possible")
        exit(-1)
    decode()
```

Running the preceding code generates the following output:

```py
Reading model parameters from /tmp/translate/translate.ckpt-10500
-------------------
Source sentence: ['wiederaufnahme', 'der', 'sitzungsperiode']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7, 618]
Ideal tokens out: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['resumption', 'of', 'the', 'session']
Model output: resumption of the session
-------------------
Source sentence: ['ich', 'bitte', 'sie', ',', 'sich', 'zu', 'einer', 'schweigeminute', 'zu', 'erheben', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 266, 22, 5, 29, 14, 78, 3931, 14, 2414, 4]
Ideal tokens out: [1, 651, 932, 6, 159, 6, 19, 11, 1440, 35, 51, 2639, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['please', 'rise', ',', 'then', ',', 'for', 'this', 'minute', "'", 's', 'silence', '.']
Model output: i ask you to move , on an approach an approach .
-------------------
Source sentence: ['(', 'das', 'parlament', 'erhebt', 'sich', 'zu', 'einer', 'schweigeminute', '.', ')']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 11, 58, 3267, 29, 14, 78, 3931, 4, 51]
Ideal tokens out: [1, 54, 5, 267, 3541, 14, 2095, 12, 1440, 35, 51, 2639, 53, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['(', 'the', 'house', 'rose', 'and', 'observed', 'a', 'minute', "'", 's', 'silence', ')']
Model output: ( the house ( observed and observed a speaker )
-------------------
Source sentence: ['frau', 'präsidentin', ',', 'zur', 'geschäftsordnung', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 151, 5, 49, 488, 4]
Ideal tokens out: [1, 212, 44, 6, 22, 12, 91, 8, 218, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['madam', 'president', ',', 'on', 'a', 'point', 'of', 'order', '.']
Model output: madam president , on a point of order .
-------------------
Source sentence: ['wenn', 'das', 'haus', 'damit', 'einverstanden', 'ist', ',', 'werde', 'ich', 'dem', 'vorschlag', 'von', 'herrn', 'evans', 'folgen', '.']
Source tokens: [0, 0, 0, 0, 85, 11, 603, 113, 831, 9, 5, 243, 13, 39, 141, 18, 116, 1939, 417, 4]
Ideal tokens out: [1, 87, 5, 267, 2096, 6, 16, 213, 47, 29, 27, 1941, 25, 1441, 4, 2, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['if', 'the', 'house', 'agrees', ',', 'i', 'shall', 'do', 'as', 'mr', 'evans', 'has', 'suggested', '.']
Model output: if the house gave this proposal , i would like to hear mr byrne .
```

As you can see, the output is mainly correct, although there are still some problematic tokens. To mitigate the problem, we'd need a more complex RNN, a longer corpora or a more diverse one.

# Home assignments

This model is trained and tested on the same dataset; that's not ideal for data science, but it was needed to have a working project. Try to find a longer corpora and split it into two pieces, one for training and one for testing:

*   Change the settings of the model: how does that impact the performance and the training time?
*   Analyze the code in `seq2seq_model.py`. How can you insert the plot of the loss in TensorBoard?
*   NLTK also contains the French corpora; can you create a system to translate them both together?


In this chapter we've seen how to create a machine translation system based on an RNN. We've seen how to organize the corpus, how to train it and how to test it. In the next chapter, we'll see another application where RNN can be used: chatbots.
