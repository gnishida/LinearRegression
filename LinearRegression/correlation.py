import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



# データの読み込み
X = []
for l in open('trueY.txt').readlines():
	list = l.split(' ')
	rec = [float(d) for d in list]
	X.append(rec)

X = np.array(X)

Y = []
for l in open('predY.txt').readlines():
	list = l.split(' ')
	rec = [float(d) for d in list]
	Y.append(rec)

Y = np.array(Y)

for i in range(X.shape[1]):
	plt.axis([-2, 2, -2, 2])
	#plt.axis('scaled')
	plt.xlabel('true')
	plt.ylabel('predicted')
	plt.plot(X[:,i], Y[:,i], 'b.', label='predicted');
	#plt.legend(loc='lower right')
	plt.savefig('correlation/correlation' + str(i) + '.png')
	plt.clf()
