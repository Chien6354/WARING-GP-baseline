import pandas as pd
import numpy

# list = [[1, 2]]
#
name = ['SVM', 'RF', 'ERF', 'LR', 'knn', 'ada']

test = pd.DataFrame(columns=name)

test.to_csv('jaffe_baseline.csv')

# results = pd.read_csv('f1.csv', index_col=0)
# print(numpy.asarray(results['lsvm']))
# a = []
# a = numpy.asarray(results['lsvm'])
# print(type(a))
# lenet_re = []
# cnn_re = []
# for i in range(5):
#     lenet_re.append(i)
# for j in range(5):
#     cnn_re.append(j)
# results['Lenet'] = lenet_re
# results['CNN'] = cnn_re
# results.to_csv('jaffe.csv')