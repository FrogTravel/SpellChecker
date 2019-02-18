import nltk
from bs4 import BeautifulSoup
from glob import glob
from nltk.metrics import edit_distance
import pickle
import os
from nltk.tokenize import RegexpTokenizer
import random
import string


def check_words(sentence, bigram_dictionary):
    """
    Check word for in-word error
    :param sentence:
    :param bigram_dictionary:
    :return:
    """
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sentence.lower())
    result = {}
    for word in words:
        try:
            result[word] = get_closest_bigrams(word, bigram_dictionary)
        except KeyError:
            pass
    return result


def get_closest_bigrams(word, bigram_dictionary):
    """
    Find the closes to word bigram that described in the paper
    :param word:
    :param bigram_dictionary:
    :return:
    """
    bigram_words = []
    for bigram in get_bigrams(word):
        try:
            bigram_words += (bigram_dictionary[bigram])
        except KeyError:
            pass

    result = []
    for bigram in bigram_words:
        result.append(bigram)

    result = get_top_from_edit_distance(word, result)

    return result


def get_top_from_edit_distance(word, suggested_words):
    """
    Based on edit distance in counts the best candidates to replace
    :param word:
    :param suggested_words:
    :return: top-10 closest to original word
    """
    top = {}

    for suggested_word in suggested_words:
        value = edit_distance(word, suggested_word)
        if value not in top:
            top[value] = [suggested_word]
        else:
            if suggested_word not in top[value]:
                top[value] += [suggested_word]

    sorted_top = dict(sorted(top.items(), key=lambda x: x[0]))

    for k, v in sorted_top.items():
        sorted_top[k] = sorted(v, key=lambda x: abs(len(word) - len(x)))

    result = []

    for elements_array in sorted_top.values():
        result += elements_array

    return result[:10]


def get_bigrams(word):
    """
    Returns all bigrams for passed word
    :param word:
    :return:
    """
    kgrams = []
    for index in range(2, len(word) + 1):
        key = word[index - 2: index]
        kgrams.append(key)
    return kgrams


def preprocess(word):
    """
    Preprocessing of word. In external method to be able to update in future
    :param word:
    :return:
    """
    return word.lower()


def build_bigram_index(path):
    """
    Build bigram index dataset to work with in future from reuters dataset. "2-gram" : [word1, word2, wordn]
    :param path:
    :return:
    """
    filenames = sorted(glob(path + 'reut2-0**.sgm'))

    index = {}
    counter = 0
    for f in filenames:
        # Чтение файлов
        reuter_stream = open(f, encoding="latin-1")
        reuter_content = reuter_stream.read()
        soup = BeautifulSoup(reuter_content, "html.parser")
        articles = soup.find_all('reuters')

        for article in articles:
            # Индексирование
            try:
                title, body = get_article(article)
                text = title + '\n' + body

                # doc_index[int(article['newid'])] = preprocess(nltk.word_tokenize(text))
                tokenizer = RegexpTokenizer(r'\w+')
                words = tokenizer.tokenize(text.lower())

                for word_full in words:
                    if len(word_full) < 2: continue
                    word = preprocess(word_full)
                    bigrams = get_bigrams(word)
                    for bigram in bigrams:
                        if bigram not in index.keys():
                            index[bigram] = [word]
                        else:
                            if word not in index[bigram]:
                                index[bigram] += [word]

                counter += 1
            except AttributeError:  # TODO уменьшить exception
                pass

    return index


def preprocess_text(text):
    """
    Preprocessing for the whole text
    :param text:
    :return:
    """
    return text.lower().replace(r'\w+', '')


def build_3_gram_index(path):
    """
    Build 3-gram index dataset to work with in future from reuters dataset. "word word word" : number of occurances
    :param path:
    :return:
    """
    filenames = sorted(glob(path + 'reut2-0**.sgm'))

    index = {}
    for f in filenames:
        # Чтение файлов
        reuter_stream = open(f, encoding="latin-1")
        reuter_content = reuter_stream.read()
        soup = BeautifulSoup(reuter_content, "html.parser")
        articles = soup.find_all('reuters')

        for article in articles:

            try:
                title, body = get_article(article)
                text = title + '\n' + body

                tokenizer = RegexpTokenizer(r'\w+')
                words = tokenizer.tokenize(text.lower())

                for counter, word_full in enumerate(words):
                    try:
                        key = word_full + " " + words[counter + 1] + " " + words[counter + 2]
                        if key not in index.keys():
                            index[key] = 1
                        else:
                            index[key] += 1
                    except IndexError:
                        pass
            except AttributeError:
                pass

    return index


def get_all_3_gram(words):
    """
    Get all 3-grams from text that is passed here
    """
    index = {}
    for counter, word_full in enumerate(words):
        try:
            key = word_full + " " + words[counter + 1] + " " + words[counter + 2]
            if key not in index.keys():
                index[words[counter + 2]] = key
            else:
                index[words[counter + 2]] += key
        except IndexError:
            pass

    return index


def context_sensitive_checking(sentence, threegram_dictionary, top_of_words):
    """
    Check errors with awareness of context
    Pseudocode and explanations for this method can be found in paper
    """
    list_of_words_from_sentence = nltk.word_tokenize(sentence)
    grams = get_all_3_gram(list_of_words_from_sentence)
    index = 0
    for key, gram in grams.items():  # Проходимся по каждой грамме
        tokenized_gram = nltk.word_tokenize(gram)
        max_number = 0
        right_word = key
        try:
            for word in top_of_words[key]:  # Проходимся по каждому предложенному слову
                checked_gram = tokenized_gram[0] + " " + tokenized_gram[1] + " " + word
                try:
                    number = threegram_dictionary[checked_gram]
                except KeyError:
                    number = 1
                if max_number < number:
                    max_number = number
                    right_word = word
        except KeyError:
            pass

        list_of_words_from_sentence[index + 2] = right_word
        index += 1
        if right_word != key:
            print("Found an error in " + key + ". Replace with " + right_word)

    # for gram in grams:

    return ' '.join(list_of_words_from_sentence)


def get_article(text):
    """
    Get article from text with beautiful soup
    """
    title = ""
    body = ""
    try:
        title = text.title.string
    except AttributeError:
        print("no TITLE: " + text['newid'])

    try:
        body = text.body.string
    except AttributeError:
        print("No BODY : " + text['newid'])

    if title == "" and body == "":
        raise AttributeError
    return title, body


def random_mistake(ngram, p=0.3):
    """
    Make a random symbol permutation with probability p
    :param ngram:
    :param p: probability of permutation
    :return: new ngram as a string
    """
    copied = list(ngram)
    word = copied[-1]

    outcome = random.random()
    if outcome <= p:
        ix = random.choice(range(len(word)))
        new_word = ''.join([word[w] if w != ix else random.choice(string.ascii_letters) for w in range(len(word))])
        copied[-1] = new_word
    else:
        copied[-1] = word

    return ' '.join(copied)


def main():
    with open('input.txt', 'r') as fp:
        sentence = fp.read()

    print("Input: " + sentence)
    print("Processing...")

    if not os.path.isfile('index.p'):
        index = build_bigram_index('reuters21578/')

        index_file = open("index.p", "wb")
        pickle.dump(index, index_file)
        index_file.close()

        file_object = open("index.txt", "w")
        file_object.write(str(index))
        file_object.close()
    if not os.path.isfile('index_sentences.p'):
        index = build_3_gram_index('reuters21578/')

        index_file = open("index_sentences.p", "wb")
        pickle.dump(index, index_file)
        index_file.close()

        file_object = open("index_sentences_txt.txt", "w")
        file_object.write(str(index))
        file_object.close()

    with open('index.p', 'rb') as fp:
        bigram_dictionary = pickle.load(fp)

    with open('index_sentences.p', 'rb') as fp:
        threegram_dictionary = pickle.load(fp)

    result_top_for_words = check_words(sentence, bigram_dictionary)
    result = context_sensitive_checking(sentence, threegram_dictionary, result_top_for_words)
    print("Result: " + result)
    file_object = open("output.txt", "w")
    file_object.write(result)
    file_object.close()

    # error = 0
    # index = 0
    # for gram in list(threegram_dictionary.keys())[:1000]:
    #     mistaked = random_mistake(gram.split(' '))
    #     result_top_for_words = check_words(mistaked, bigram_dictionary)
    #     result = context_sensitive_checking(mistaked, threegram_dictionary, result_top_for_words)
    #     if gram != result:
    #         error += 1
    #     print("Total: " + str(index) + " Errors: " + str(error))
    #     index += 1
    #
    # print(error)

if __name__ == "__main__":
    main()
