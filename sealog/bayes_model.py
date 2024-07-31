import math
from collections import defaultdict
from common.utils import calculate_metrics


class NaiveBayesAgent:
    def __init__(self, n=1):
        self.class_counts = defaultdict(int)
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.ngram_counts_reverse = defaultdict(lambda: defaultdict(int))
        self.total_documents = 0
        self.n = n

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            ngrams = self.generate_ngrams(doc)
            for ngram in ngrams:
                self.ngram_counts[ngram][label] += 1
                self.ngram_counts_reverse[label][ngram] += 1
            self.total_documents += 1

    def predict(self, document, threshold=0.001):
        max_prob = -math.inf
        predicted_class = None

        ngrams = self.generate_ngrams(document)

        label_prob = {}
        for label in self.class_counts.keys():
            log_prob = math.log(self.class_counts[label] / self.total_documents)

            for ngram in ngrams:
                if self.ngram_counts[ngram][label] == 0 and label == 1 and threshold is not None:
                    # emphasize the anomaly ngrams
                    max_anomaly_count = sum(self.ngram_counts_reverse[label].values()) / len(self.ngram_counts_reverse[label])
                    ngram_prob = (max_anomaly_count + 1)  / (sum(self.ngram_counts[ngram].values()) + 1 * len(self.ngram_counts))
                else:
                    alpha = 1e-5
                    ngram_prob = (self.ngram_counts[ngram][label] + alpha) / (sum(self.ngram_counts[ngram].values()) + alpha * len(self.ngram_counts))
                log_prob += math.log(ngram_prob)

            label_prob[label] = log_prob
            
            if log_prob > max_prob:
                max_prob = log_prob
                predicted_class = label
    
        return predicted_class
    
    def evaluate_samples(self, documents, labels):
        predicted_labels = []
        for doc in documents:
            predicted_labels.append(self.predict(doc))
        return calculate_metrics(labels, predicted_labels)

    def update(self, document, label):
        self.class_counts[label] += 1
        ngrams = self.generate_ngrams(document)
        for ngram in ngrams:
            self.ngram_counts[ngram][label] += 1
            self.ngram_counts_reverse[label][ngram] += 1
        self.total_documents += 1

    def generate_ngrams(self, document):
        ngrams = []
        for i in range(len(document) - self.n + 1):
            ngram = " ".join(document[i:i+self.n])
            ngrams.append(ngram)
        return ngrams