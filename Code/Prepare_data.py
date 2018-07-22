import os
import numpy as np
from matplotlib import pyplot as plt

def main():
	folder_name = '../Data/Mails/'
	files = []
	for f in os.listdir(folder_name):
		files.append(folder_name + f)

	if1 = '../Data/Data_spam.txt'
	f1 = open(if1, 'w')
	if2 = '../Data/Data_ham.txt'
	f2 = open(if2, 'w')
	if3 = '../Data/Data.txt'
	f3 = open(if3, 'w')
	if4 = '../Data/Labels.txt'
	f4 = open(if4, 'w')
	count_ham = 0
	count_spm = 0
	for i in files:
		f = open(i, 'r')
		line1 = f.readline()[0:-1]
		line1 = line1.split(': ')[1]
		line2 = f.readline()
		line2 = f.readline()[0:-1]
		d = line1 + ' ' + line2
		if 'legit' in i:
			f1.write(d + '\n')
			f4.write('0' + '\n')
			count_ham += 1
		else:
			f2.write(d + '\n')
			f4.write('1' + '\n')
			count_spm += 1
		f3.write(d + '\n')
		f.close()
	f1.close()
	f2.close()
	f3.close()
	f4.close()

	print('Getting class counts')
	print('# Ham mails:', count_ham)
	print('# Spam mails:', count_spm)
	count = count_ham + count_spm

	data = []
	labels = []
	with open('../Data/Data.txt') as f:
		for line in f:
			data.append(line[0:-1])
	with open('../Data/Labels.txt') as f:
		for line in f:
			labels.append(line[0:-1])

	for i in range(1, 6):
		if1 = '../Data/Set' + str(i) + '/Train_x.txt'
		f1 = open(if1, 'w')
		if2 = '../Data/Set' + str(i) + '/Train_y.txt'
		f2 = open(if2, 'w')
		if3 = '../Data/Set' + str(i) + '/Test_x.txt'
		f3 = open(if3, 'w')
		if4 = '../Data/Set' + str(i) + '/Test_y.txt'
		f4 = open(if4, 'w')
		perm = np.random.permutation(count)
		count_ham = 0
		count_spm = 0
		for j in range(850):
			f1.write(data[perm[j]] + '\n')
			f2.write(labels[perm[j]] + '\n')
			if labels[perm[j]] == '0':
				count_ham += 1
			else:
				count_spm += 1
		print('(Train) Getting class counts for set', i)
		print('# Ham mails:', count_ham)
		print('# Spam mails:', count_spm)

		count_ham = 0
		count_spm = 0
		for j in range(850, count):
			f3.write(data[perm[j]] + '\n')
			f4.write(labels[perm[j]] + '\n')
			if labels[perm[j]] == '0':
				count_ham += 1
			else:
				count_spm += 1
		print('(Test) Getting class counts for set', i)
		print('# Ham mails:', count_ham)
		print('# Spam mails:', count_spm)
		f1.close()
		f3.close()
		f2.close()
		f4.close()

if __name__ == '__main__':
	main()
