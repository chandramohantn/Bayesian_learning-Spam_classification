import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from math import gamma


class Spam_classifier:

	def __init__(self, spam_x, ham_x, vocab):
		self.spam_x = spam_x
		self.ham_x = ham_x
		self.model = {'vocab': vocab, 'params': {}, 'model_perf': {}}

	def mle_multinomial_train(self):
		params = {}
		tot = np.array([0.0, 0.0])
		for k in self.model['vocab'].keys():
			params[k] = self.model['vocab'][k] + 1.0
			tot += params[k]
		for k in self.model['vocab'].keys():
			params[k] = params[k] / tot
		self.model['params'] = params

		score, labels = self.mle_multinomial_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def mle_multinomial_test(self, data):
		score = np.zeros((len(data), 2))
		for i in range(len(data)):
			items = data[i].split(' ')
			for w in items:
				if w in self.model['vocab']:
					score[i] += np.log(self.model['params'][w])
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def map_multinomial_dirichlet_train(self, alpha):
		params = {}
		prior = {}
		tot = np.array([0.0, 0.0])

		if '__all__' in alpha:
			for k in self.model['vocab'].keys():
				prior[k] = alpha['__all__']
		else:
			for k in alpha.keys():
				if k == '__rest__':
					for ky in self.model['vocab'].keys():
						prior[ky] = alpha['rest']
			for k in alpha.keys():
				if k != '__rest__':
					prior[k] = alpha[k]
		for k in self.model['vocab'].keys():
			params[k] = self.model['vocab'][k] + prior[k] - 1.0
			tot += params[k]
		for k in self.model['vocab'].keys():
			params[k] = params[k] / tot
		self.model['prior'] = prior
		self.model['params'] = params

		score, labels = self.map_multinomial_dirichlet_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def map_multinomial_dirichlet_test(self, data):
		score = np.zeros((len(data), 2))
		for i in range(len(data)):
			items = data[i].split(' ')
			for w in items:
				if w in self.model['vocab']:
					score[i] += np.log(self.model['params'][w])
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def bayesian_multinomial_dirichlet_train(self, alpha):
		params = {}
		prior = {}
		tot = np.array([0.0, 0.0]).reshape((1, 2))

		if '__all__' in alpha:
			for k in self.model['vocab'].keys():
				prior[k] = alpha['__all__']
		else:
			for k in alpha.keys():
				if k == '__rest__':
					for ky in self.model['vocab'].keys():
						prior[ky] = alpha['rest']
			for k in alpha.keys():
				if k != '__rest__':
					prior[k] = alpha[k]
		for k in self.model['vocab'].keys():
			tot += (self.model['vocab'][k] + prior[k])
		for k in self.model['vocab'].keys():
			params[k] = (self.model['vocab'][k] + prior[k]) / tot
		self.model['prior'] = prior
		self.model['params'] = params

		score, labels = self.bayesian_multinomial_dirichlet_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def bayesian_multinomial_dirichlet_test(self, data):
		score = []
		for i in range(len(data)):
			items = data[i].split(' ')
			s = np.array([0.0, 0.0]).reshape((1, 2))
			for w in items:
				if w in self.model['vocab']:
					s += np.log(self.model['params'][w])
			score.append(s)
		score = np.array(score).reshape((len(data), 2))
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def mle_bernoulli_train(self):
		params = {}
		s = np.array([len(self.ham_x), len(self.spam_x)]) + 2.0
		for k in self.model['vocab'].keys():
			params[k] = (self.model['vocab'][k] + 1.0) / s
		self.model['params'] = params

		score, labels = self.mle_bernoulli_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def mle_bernoulli_test(self, data):
		score = np.zeros((len(data), 2))
		for i in range(len(data)):
			items = data[i].split(' ')
			for k in self.model['vocab'].keys():
				if k in items:
					score[i] += np.log(self.model['params'][k])
				else:
					score[i] += np.log(1.0 - self.model['params'][k])
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def map_bernoulli_beta_train(self, alpha, beta):
		params = {}
		s1 = np.array([alpha, beta])
		s2 = np.array([len(self.ham_x), len(self.spam_x)]) + alpha + beta - 2.0
		for k in self.model['vocab'].keys():
			params[k] = (self.model['vocab'][k] + s1 - 1.0) / s2
		self.model['prior'] = np.array([alpha, beta]).reshape((1, 2))
		self.model['params'] = params

		score, labels = self.map_bernoulli_beta_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def map_bernoulli_beta_test(self, data):
		score = np.zeros((len(data), 2))
		for i in range(len(data)):
			items = data[i].split(' ')
			for k in self.model['vocab'].keys():
				if k in items:
					score[i] += np.log(self.model['params'][k])
				else:
					score[i] += np.log(1 - self.model['params'][k])
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def bayesian_bernoulli_beta_train(self, alpha, beta):
		params = {}
		s1 = np.array([alpha, beta])
		s2 = np.array([len(self.ham_x), len(self.spam_x)]) + alpha + beta
		for k in self.model['vocab'].keys():
			params[k] = (self.model['vocab'][k] + s1) / s2
		self.model['prior'] = np.array([alpha, beta]).reshape((1, 2))
		self.model['params'] = params

		score, labels = self.bayesian_bernoulli_beta_test(np.concatenate((self.ham_x, self.spam_x)))
		self.model_performance(labels, score)
		print('Train Performance: ', self.model['model_perf'])

	def bayesian_bernoulli_beta_test(self, data):
		score = np.zeros((len(data), 2))
		for i in range(len(data)):
			items = data[i].split(' ')
			for k in self.model['vocab'].keys():
				if k in items:
					score[i] += np.log(self.model['params'][k])
				else:
					score[i] += np.log(1 - self.model['params'][k])
		labels = np.argmax(score, axis=1)
		score = np.max(score, axis=1)
		return score, labels

	def model_performance(self, pred_y, score):
		true_y = np.concatenate((np.zeros((len(self.ham_x), 1)), np.ones((len(self.spam_x), 1))))
		true_y = np.reshape(true_y, (len(true_y), 1))
		precision, recall, thresholds = precision_recall_curve(true_y, score)
		auc = metrics.auc(recall, precision, reorder=True)
		acc = 0.0
		for i in range(len(true_y)):
			if true_y[i][0] == pred_y[0]:
				acc += 1
		acc = acc / true_y.shape[0]
		self.model['model_perf'] = {'acc': acc, 'auc': auc}

	def save_model(self, filename):
		with open(filename + '.pickle', 'wb') as f:
			pickle.dump(self.model, f)

