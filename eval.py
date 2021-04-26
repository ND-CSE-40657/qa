from collections import Counter
import string
import re

# From official SQuAD evaluate-1.1.py

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def remove_bpe(s):
    return s.replace('@@ ', '')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("usage: eval.py <your-output> <gold-data>", file=sys.stderr)
        print(file=sys.stderr)
        print("<your-output>: one answer per line")
        print("<gold-data>: tab-separated context, question, answer spans")
        sys.exit()

    test = []
    for line in open(sys.argv[1]):
        test.append(remove_bpe(line))
    
    gold = []
    for line in open(sys.argv[2]):
        context, question, answers_string = line.split('\t')
        context = context.split()
        answers = set()
        for span in answers_string.split():
            span = span.split('-')
            start = int(span[0])
            end = int(span[1])
            answers.add(remove_bpe(' '.join(context[start:end])))
        gold.append(answers)

    total = n = 0
    for testline, goldline in zip(test, gold):
        total += metric_max_over_ground_truths(f1_score, testline, goldline)
        n += 1
    print(total/n)
