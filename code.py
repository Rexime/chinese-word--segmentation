import argparse
import codecs
import sys
from collections import Counter, defaultdict

import numpy as np


def make_label(text):  # 对每个分词进行标注
    output = []
    if len(text) == 1:
        output.append('S')
    else:
        output += ['B'] + ['M'] * (len(text) - 2) + ['E']
    return output


def pre_pro(path):  # 给训练集进行标注
    label = []
    text = []
    sentence_cnt = 0
    with open(path, encoding='utf-8') as file:
        for line in file:
            sentence_cnt += 1
            line = line.strip()
            words = line.split()
            temp_label = []
            temp_str = []
            for word in words:
                temp_label += make_label(word)
                temp_str += word
            label.append(temp_label)
            text.append(temp_str)
    return text, label, sentence_cnt


def gen_words(path):
    text = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words = line.split()
            temp_str = []
            for word in words:
                temp_str += word
            text.append(temp_str)
    return text


class StructuredPerceptron(object):

    def __init__(self):

        self.feature_weights = defaultdict(float)
        self.label_type = ['B', 'M', 'E', 'S']
        self.begin = '#'
        self.end = '#'

    def get_features(self, word, word_l1, word_l2, word_r1, word_r2, tag_l1, tag):
        features = ['1' + word,
                    '2' + word_l1,
                    '3' + word_r1,
                    '4' + word_l2 + word_l1,
                    '5' + word_l1 + word,
                    '6' + word + word_r1,
                    '7' + word_r1 + word_r2,
                    '8' + word_l1 + word + word_r1,
                    '9' + word + tag,
                    '11' + tag_l1 + tag,
                    '12' + word_l1 + tag,
                    '13' + word_l2 + word_l1 + tag,
                    '14' + word_r1 + tag,
                    '15' + word_l1 + word + tag,
                    '16' + word + word_r1 + tag,
                    '10' + word_r1 + word_r2 + tag
                    ]
        return features

    def get_all_features(self, sentence, labels):
        all_features = Counter()
        length = len(sentence)
        for i in range(length):
            word_l2 = sentence[i - 2] if i - 2 >= 0 else '#'
            word_l1 = sentence[i - 1] if i - 1 >= 0 else '#'
            word = sentence[i]
            word_r1 = sentence[i + 1] if i + 1 < len(sentence) else '#'
            word_r2 = sentence[i + 2] if i + 2 < len(sentence) else '#'
            tag_l1 = labels[i - 1] if i - 1 >= 0 else '#'
            tag = labels[i]
            all_features.update(self.get_features(word, word_l1, word_l2, word_r1, word_r2, tag_l1, tag))
        return all_features

    def decode(self, sentence):  # 预测

        length = len(sentence)
        if length == 0:
            return []
        elif length == 1:
            best_label = ['S']
        else:
            score = np.ones((4, length), dtype=float) * float('-Inf')
            path = np.ones((4, length), dtype=int) * -1

            # 初始化
            word = sentence[0]
            word_l1 = '#'
            word_l2 = '#'
            word_r1 = sentence[1] if 1 < len(sentence) else '#'
            word_r2 = sentence[2] if 2 < len(sentence) else '#'
            for i in range(4):
                features = self.get_features(word, word_l1, word_l2, word_r1, word_r2, '#', self.label_type[i])
                score[i][0] = sum(self.feature_weights[x] for x in features)

            for i in range(1, length):  # 每个字
                word_l2 = sentence[i - 2] if i - 2 >= 0 else '#'
                word_l1 = sentence[i - 1] if i - 1 >= 0 else '#'
                word = sentence[i]
                word_r1 = sentence[i + 1] if i + 1 < len(sentence) else '#'
                word_r2 = sentence[i + 2] if i + 2 < len(sentence) else '#'
                for j in range(4):  # 该字的标签

                    tag_ = self.label_type[j]

                    for k in range(4):  # 前一个字的标签

                        tag_l1 = self.label_type[k]
                        features = self.get_features(word, word_l1, word_l2, word_r1, word_r2, tag_l1, tag_)

                        temp_score = sum(self.feature_weights[mk] for mk in features)
                        if temp_score + score[k][i - 1] > score[j][i]:
                            score[j][i] = temp_score + score[k][i - 1]
                            path[j][i] = k

            best_path = [0, ]
            best_score = float('-Inf')
            for i in range(4):
                if score[i][length - 1] > best_score:
                    best_score = score[i][length - 1]
                    best_path[0] = i
            pos = length - 1
            next_ = best_path[0]
            while pos > 0:
                best_path.append(path[next_][pos])
                next_ = path[next_][pos]
                pos -= 1

            best_label = []
            for i in range(length):
                best_label.append(self.label_type[best_path[length - 1 - i]])

        return best_label

    def fit(self, sentence_cnt, x, y, iterations=5):
        weights = Counter()
        counter = 0
        for i in range(iterations):
            correct = 0
            total = 0
            for j in range(sentence_cnt):  # 每一行句子
                counter += 1
                predict_label = self.decode(x[j])
                all_features = self.get_all_features(x[j], predict_label)
                all_gold_features = self.get_all_features(x[j], y[j])

                for fid, count in all_gold_features.items():
                    self.feature_weights[fid] += count
                for fid, count in all_features.items():
                    self.feature_weights[fid] -= count

                correct += sum([1 for (predicted, gold) in zip(predict_label, y[j]) if predicted == gold])
                total += len(y[j])
                if counter % 1000 == 0:
                    print(counter)
                    print('\tTraining accuracy: %.4f\n\n' % (correct / total))

            weights.update(self.feature_weights)

        self.feature_weights = weights
        return self.feature_weights

    def predict_(self, testdata):
        predict_sentence = []
        for line in testdata:
            predict_tag = self.decode(line)
            this_sentence = []
            lth = len(line)
            for i in range(lth):
                if predict_tag[i] == 'S' or predict_tag[i] == 'E':
                    this_sentence.append(line[i])
                    this_sentence.append(' ')
                elif predict_tag[i] == 'B' or predict_tag[i] == 'M':
                    this_sentence.append(line[i])
            predict_sentence.append(this_sentence)
        return predict_sentence

    def load_model(self, file_name):
        with open(file_name, 'r') as fp:
            for line in fp.readlines():
                data1 = line.split(":")[0]
                data2 = line.split(":")[1]
                self.feature_weights[data1] = float(data2)

    def save(self, text, file_name):
        with codecs.open(file_name, "w", encoding='utf-8') as model:
            for line in text:
                line.append('\n')
                model.writelines(line)


if __name__ == "__main__":

    SP = StructuredPerceptron()

    words = gen_words('dev.txt')
    SP.save(words, 'words.txt')

    # train
    train_sentences, gold_tags, lines_cnt = pre_pro('train.txt')

    # iterations=5
    feature_weights = SP.fit(lines_cnt, train_sentences, gold_tags, 5)
    # save the model
    target_f = open('model3.txt', 'w', encoding='utf-8')
    for key in feature_weights.keys():
        target_f.writelines(key + ":" + str(feature_weights[key]) + '\n')
    print('write done')
    # SP.load_model('model3.txt')
    # print('load succeed')

    # predict
    train_sentences, gold_tags, lines_cnt = pre_pro('dev.txt')
    predict_text = SP.predict_(train_sentences)
    SP.save(predict_text, 'predict3.txt')

    # 5 more times, iterations=10
    feature_weights = SP.fit(lines_cnt, train_sentences, gold_tags, 5)
    target_f = open('model3_1.txt', 'w', encoding='utf-8')
    for key in feature_weights.keys():
        target_f.writelines(key + ":" + str(feature_weights[key]) + '\n')
    print('write done')

    # predict
    train_sentences, gold_tags, lines_cnt = pre_pro('dev.txt')
    predict_text = SP.predict_(train_sentences)
    SP.save(predict_text, 'predict3_1.txt')

    # make prediction for test
    train_sentences, gold_tags, lines_cnt = pre_pro('test.txt')
    predict_text = SP.predict_(train_sentences)
    SP.save(predict_text, 'result.txt')
    print('predict succeed')
