import pickle

from deap import gp


def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return



def saveLog (fileName, log):
   f = open(fileName, 'wb')
   pickle.dump(log, f)
   f.close()
   return


def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        

def saveAllResults(randomSeeds, dataSetName, hof, trainTime, testResults, log, f1):
    fileName1 = str(randomSeeds) + 'Results_on' + dataSetName + '.txt'
    saveLog(fileName1, log)
    fileName = str(randomSeeds) + 'Final_Result_son' + dataSetName + '.txt'
    saveResults(fileName, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', hof[0].fitness,
                         'testResults', testResults, 'bestInd in training',
                         hof[0], 'F1-measure', f1)

    return
