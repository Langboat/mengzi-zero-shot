from __future__ import division
from sklearn.metrics import accuracy_score, f1_score
import collections
import six


def cal_acc(labels, preds):
    return accuracy_score(labels, preds)


def cal_f1(labels, preds):
    return f1_score(labels, preds, average='micro')


def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(pre_lines, gold_lines):

        pre = pre.get(label, {})
        gold = gold.get(label, {})

        for i in pre:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pre:
                FN += 1

    # print(TP, FP, FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)
    # print(p, r, f)
    return f


def ner_get_f1_score(pre_lines=None, gold_lines=None, labels=None):
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]

    f_score = {}

    # print(sum_num([ list(gold.keys()) for gold in gold_lines],[]))

    gold_labels = set(sum([list(gold.keys()) for gold in gold_lines], []))
    pre_labels = set(sum([list(pre.keys()) for pre in pre_lines], []))
    # print(gold_labels,pre_labels)

    sum_num = 0
    for label in labels:
        if label not in gold_labels or label not in pre_labels:
            continue
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum_num += f

    avg = sum_num / len(labels)
    return f_score, avg


def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)


def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))


def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)


def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result


def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def _safe_f1(matches, recall_total, precision_total, alpha=0.5):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0


def rouge_n(peer, model, n):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    peer_counter = _ngram_counts(peer, n)
    '''
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    '''
    model_counter = _ngram_counts(model, n)
    matches = _counter_overlap(peer_counter, model_counter)
    recall_total = _ngram_count(model, n)
    precision_total = _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total)


def rouge_2_multiple_target(peers, model):
    """
    Compute the ROUGE-2 (bigram) score of a list of peers with respect to one model.
    """
    rouge_2 = 0
    for peer in peers:
        rouge_2 += rouge_n(peer, model, 2)
    rouge_2 /= len(peers)
    return rouge_2


def rouge_2_corpus_multiple_target(peers, models):
    curpus_size = len(peers)
    rouge_score = 0
    for (peer, model) in zip(peers, models):
        rouge_score += rouge_2_multiple_target(peer, model)
    return rouge_score / curpus_size
