import numpy as np
import sys
import pickle
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn import metrics
from Spam_classifier import Spam_classifier

def get_data(i_f):
	data = []
	with open(i_f) as f:
		for line in f:
			data.append(line[0:-1])
	return data

def performance(true_y, pred_y, score):
	precision, recall, thresholds = precision_recall_curve(true_y, score)
	auc = metrics.auc(recall, precision, reorder=True)
	acc = np.sum(true_y == pred_y)
	acc = acc / len(true_y)
	return acc, auc

def multinomial_count(data1, data2):
	vocab = {}
	for line in data1:
		items = line.split(' ')
		for i in items:
			if i not in vocab:
				vocab[i] = np.array([0, 1])
			else:
				vocab[i][1] += 1
	for line in data2:
		items = line.split(' ')
		for i in items:
			if i not in vocab:
				vocab[i] = np.array([1, 0])
			else:
				vocab[i][0] += 1
	return vocab

def bernoulli_count(data1, data2):
	vocab = {}
	data = []
	for line in data1:
		items = list(set(line.split(' ')))
		data.append(items)
		for w in items:
			if w not in vocab:
				vocab[w] = np.array([0, 0])
	for line in data2:
		items = list(set(line.split(' ')))
		data.append(items)
		for w in items:
			if w not in vocab:
				vocab[w] = np.array([0, 0])
	l = len(data1)

	for k in vocab.keys():
		for i in range(l):
			if k in data[i]:
				vocab[k][1] += 1
		for i in range(l, len(data)):
			if k in data[i]:
				vocab[k][0] += 1
	return vocab

def main():
	print('Loading train data ...')
	i_f = '../Data/' + sys.argv[1] + '/Train_x.txt'
	train_x = get_data(i_f)
	i_f = '../Data/' + sys.argv[1] + '/Train_y.txt'
	train_y = get_data(i_f)
	labels = np.array(train_y)
	spam_idx = np.where(labels == '1')[0]
	ham_idx = np.where(labels == '0')[0]
	spam_x = []
	for i in spam_idx:
		spam_x.append(train_x[i])
	ham_x = []
	for i in ham_idx:
		ham_x.append(train_x[i])

	print('Loading test data ...')
	i_f = '../Data/' + sys.argv[1] + '/Test_x.txt'
	test_x = get_data(i_f)
	i_f = '../Data/' + sys.argv[1] + '/Test_y.txt'
	test_y = get_data(i_f)
	test_y = np.array([int(i) for i in test_y])
	test_y = np.reshape(test_y, (len(test_y), 1))
	'''
	print('Building dictionaries ...')
	m_vocab = multinomial_count(spam_x, ham_x)
	with open('../model/multlinomial_vocab.pickle', 'wb') as f:
		pickle.dump(m_vocab, f)
	print('multlinomial vocabulary created ...')
	b_vocab = bernoulli_count(spam_x, ham_x)
	with open('../model/bernoulli_vocab.pickle', 'wb') as f:
		pickle.dump(b_vocab, f)
	print('bernoulli vocabulary created ...')
	'''
	print('Loading dictionaries')
	with open('../model/multlinomial_vocab.pickle', 'rb') as f:
		m_vocab = pickle.load(f)
	print('multlinomial vocabulary loaded ...')
	with open('../model/bernoulli_vocab.pickle', 'rb') as f:
		b_vocab = pickle.load(f)
	print('bernoulli vocabulary loaded ...')
	'''
	print('Training mle multlinomial model ...')
	classifier = Spam_classifier(spam_x, ham_x, m_vocab)
	classifier.mle_multinomial_train()
	classifier.save_model('../model/mle_multinomial')
	print('testing mle multlinomial')
	score, labels = classifier.mle_multinomial_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)

	print('Training map multlinomial dirichlet model ...')
	classifier = Spam_classifier(spam_x, ham_x, m_vocab)
	classifier.map_multinomial_dirichlet_train({'__all__': 2})
	classifier.save_model('../model/map_multinomial_dirichlet')
	print('testing map multlinomial dirichlet')
	score, labels = classifier.map_multinomial_dirichlet_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)

	print('Training bayesian multlinomial dirichlet model ...')
	classifier = Spam_classifier(spam_x, ham_x, m_vocab)
	classifier.bayesian_multinomial_dirichlet_train({'__all__': 2})
	classifier.save_model('../model/bayesian_multinomial_dirichlet')
	print('testing bayesian multlinomial dirichlet')
	score, labels = classifier.bayesian_multinomial_dirichlet_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)

	print('Training mle bernoulli model ...')
	classifier = Spam_classifier(spam_x, ham_x, b_vocab)
	classifier.mle_bernoulli_train()
	classifier.save_model('../model/mle_bernoulli')
	print('testing mle bernoulli')
	score, labels = classifier.mle_bernoulli_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)

	print('Training map bernoulli beta model ...')
	classifier = Spam_classifier(spam_x, ham_x, b_vocab)
	classifier.map_bernoulli_beta_train(2, 3)
	classifier.save_model('../model/map_bernoulli_beta')
	print('testing map bernoulli beta')
	score, labels = classifier.map_bernoulli_beta_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)
	'''

	print('Training bayesian bernoulli beta model ...')
	classifier = Spam_classifier(spam_x, ham_x, b_vocab)
	classifier.bayesian_bernoulli_beta_train(2, 3)
	classifier.save_model('../model/bayesian_bernoulli_beta')
	print('testing bayesian bernoulli beta')
	score, labels = classifier.bayesian_bernoulli_beta_test(test_x)
	acc, auc = performance(test_y, labels, score)
	print('Accuracy: ', acc)
	print('AUC: ', auc)

if __name__ == '__main__':
	main()

