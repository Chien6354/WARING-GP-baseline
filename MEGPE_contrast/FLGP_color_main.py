# python packages
import random
import time
import operator
import evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import numpy as np
# deap package
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Img, Region, Vector, Vector1
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import saveFile
import sys
from sklearn import metrics

'FLGP'

# randomSeeds = 1
# dataSetName = 'DSLR'
dataSetName = str(sys.argv[1])
randomSeeds = str(sys.argv[2])
dataset_path = '/nesi/project/vuw03291/cifar10_small/'

x_train = np.load(dataset_path+dataSetName+'_train_data3_1280.npy')/255.0
y_train = np.load(dataset_path+dataSetName+'_train_label_1280.npy')
x_test = np.load(dataset_path+dataSetName+'_test_data3.npy')/255.0
y_test = np.load(dataset_path+dataSetName+'_test_label.npy')

# # three channels ----rgb
# x_train = x_train[:, 0:3, :, :]
# x_test = x_test[:, 0:3, :, :]

print('x_train.shape:', x_train.shape)
print('x_train[0][0].shape:', x_train[0][0].shape)

# parameters:
population = 100
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8

bound1, bound2 = x_train[0][1, :, :].shape
##GP

pset = gp.PrimitiveSetTyped('MAIN', [Img, Img, Img], Vector1, prefix='Image')
#Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector], Vector1, name='FeaCon')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='FeaCon2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# Region detection operators
pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)

#fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)


def evalTrain(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    lsvm = LinearSVC(max_iter=100)
    accuracy = round(cross_val_score(lsvm, train_norm, y_train, cv=3).mean(), 2)
    return accuracy,

# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(randomSeeds):
    random.seed(randomSeeds)
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields
    pop, log, best_ind_gen = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof


def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j][0, :, :], x_test[j][1, :, :], x_test[j][2, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm= LinearSVC()
    lsvm.fit(train_norm, y_train)
    accuracy = round(100*lsvm.score(test_norm, y_test), 2)
    predict_labels = lsvm.predict(test_norm)
    # wrong = 0
    # for i in range(len(test_norm)):
    #     if predict_labels[i] != y_test[i]:
    #         wrong = wrong + 1
    # f3 = metrics.precision_score(y_test, predict_labels, average='weighted')  ##### precision
    # f4 = metrics.recall_score(y_test, predict_labels, average='weighted')  ##### recall
    f1 = metrics.f1_score(y_test, predict_labels, average='macro')  ##### f1-measure
    return train_tf.shape[1], accuracy, f1


if __name__ == "__main__":
    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime
    num_features, testResults, f1 = evalTest(hof[0])
    endTime1 = time.process_time()
    testTime = endTime1 - endTime
    saveFile.saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log, f1)
    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
