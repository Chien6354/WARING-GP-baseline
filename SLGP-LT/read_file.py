import os
import numpy
import stat_test
import pandas as pd



def load_data(dataset):
    # data_folder = str(method)+'/'+str(dataset)
    data_folder = str(dataset)
    data = os.listdir(data_folder)
    if '.DS_Store' in data:
        data.remove('.DS_Store')
    data_num = len(data)
    accuracy = []
    for i in range(data_num):
        with open(data_folder+'/'+data[i]) as fr:
                line = fr.readlines()[9]
                length = len(line)
                accuracy_item = float(line[0:length-1])
                print(accuracy_item)
                accuracy.append(accuracy_item)
    accuracy = numpy.array(accuracy)
    return accuracy


# def load_data(dataset):
#     # data_folder = str(method)+'/'+str(dataset)
#     data_folder = str(dataset)
#     data = os.listdir(data_folder)
#     if '.DS_Store' in data:
#         data.remove('.DS_Store')
#     data_num = len(data)
#     accuracy = []
#     tree_size = []
#     for i in range(data_num):
#         count = 0
#         with open(data_folder+'/'+data[i]) as fr:
#                 line = fr.readlines()[13]
#                 length = len(line)
#                 for j in range(0, length):
#                     if line[j] == '(' or line[j] == ')':
#                         count = count + 1
#                 print('()', count)
#                 accuracy.append(count)
#
#     for k in range(data_num):
#         with open(data_folder+'/'+data[k]) as fr:
#             line1 = fr.readlines()[17]
#             length1 = len(line1)
#             accuracy_item = float(line1[0:length1 - 1])
#             print('tree size', accuracy_item)
#             tree_size.append(accuracy_item)
#
#     accuracy = numpy.array(accuracy)
#     tree_size = numpy.array(tree_size)
#     return accuracy


def load_data_cnn(method, dataset):
    data_folder = str(method)+'/'+str(dataset)
    data = os.listdir(data_folder)
    if '.DS_Store' in data:
        data.remove('.DS_Store')
    data_num = len(data)
    accuracy = []
    for i in range(data_num):
        with open(data_folder+'/'+data[i]) as fr:
                line = fr.readlines()[11]
                # length = len(line)
                accuracy_item = float(line)

                accuracy.append(accuracy_item)
    accuracy = numpy.array(accuracy)

    return accuracy


if __name__ == "__main__":
    # accuracy1 = load_data('2T_fractal_LBP_sobel_prewitt', 'coil')
    # # accuracy2 = load_data_cnn('lenet_5_run', 'coil')
    # accuracy1 = load_data('f1')
    # mean1 = numpy.mean(accuracy1)
    # std1 = numpy.std(accuracy1)
    # max1 = numpy.max(accuracy1)
    # print('mean data1:', mean1)
    # print('std data1:', std1)
    # print('max data1:', max1)

    accuracy = load_data('f1')
    mean2 = numpy.mean(accuracy)
    std2 = numpy.std(accuracy)
    max2 = numpy.max(accuracy)
    print(len(accuracy))
    print('mean data2:', mean2)
    print('std data2:', std2)
    print('max data2', max2)

    # results = pd.read_csv('jaffe.csv', index_col=0)
    # a = []
    # a = numpy.asarray(results['Lenet'])
    # mean = numpy.mean(a)
    # std = numpy.std(a)
    # max = numpy.max(a)
    # print('mean data:', mean)
    # print('std data:', std)
    # print('max data', max)
    # print('length', len(a))
    # print('a', a)
    # print('proposed', accuracy)
    # print(accuracy1)
    # print(accuracy1[0])
    #
    # a = []
    # for i in range(len(accuracy)):
    #     a.append(83.80)

    # t_test = stat_test.wilcoxonT_unpaired(accuracy, a)
    # print(t_test)
