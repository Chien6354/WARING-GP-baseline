import pickle


def saveResults(fileName,*args,**kwargs):
    f=open(fileName,'w')
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return


def saveLog(fileName,log):
    f=open(fileName,'wb')
    pickle.dump(log,f)
    f.close()
    return


def bestInd(toolbox,population,number):
    bestInd=[]
    best=toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd
        

def saveAllResults(randomSeeds,dataSetName,best_ind_va,log,hof,num_features,trainTime,testTime,testResults):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', hof[0].fitness, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                         hof[0], 'Best individual in each run',
                        *best_ind_va[:],'final best fitness',best_ind_va[-1].fitness,
                        'initial fitness', best_ind_va[0].fitness)

    return

def saveFLGPMT4(randomSeeds,dataSetName,log,global_hof, local_hof,num_features,trainTime,testTime,testResults):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', local_hof[0].fitness, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                         global_hof[0], local_hof[0], 'Best individual of global population',
                        *global_hof[:], 'Best individual of local population',
                        *local_hof[:], 'best global fitness',global_hof[-1].fitness,
                        'best local fitness', local_hof[0].fitness)
    return

def saveFLGPMT11(randomSeeds,dataSetName,log,global_hof, local_hof,num_features,trainTime,testTime,testResults):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', local_hof[0].fitness, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                         global_hof[0], local_hof[0], 'Best individual of global population',
                        *global_hof[:], 'Best individual of local population',
                        *local_hof[:], 'best global fitness',global_hof[0].fitness,
                        'best local fitness', local_hof[0].fitness)
    return

def savettflgp(randomSeeds,dataSetName,log,global_hof, local_hof,num_features,trainTime,testTime,testResults,testResults_global,testResults_local):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', local_hof[0].fitness, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults, 'testResults Using global features', testResults_global,
                'testResults Using local features', testResults_local, 'bestInd in training',
                         global_hof[0], local_hof[0], 'Best individual of global population',
                        *global_hof[:], 'Best individual of local population',
                        *local_hof[:], 'best global fitness',global_hof[0].fitness,
                        'best local fitness', local_hof[0].fitness)
    return

def savettMT(randomSeeds,dataSetName,log,best_pop,pop, best_train,num_features,trainTime,testTime,testResults, f1):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                         'trainResults', best_train, 'number features', num_features,
                         'testTime', testTime, 'testResults', testResults,'bestInd in training (global and local)',
                         best_pop[0], best_pop[1],'bestInd in training (global )',*pop[:][0],
                'bestInd in training (local )', *pop[:][1], 'F1-measure', f1)
    return


def saveAllResults_Ensemble(randomSeeds,dataSetName, best_ind_va,log,hof,num_features,trainTime,testTime,testResults, all_acc, all_accTop, all_ensemble, all_topFiveE):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime, 'trainResults',
                hof[0].fitness, 'number features', num_features, 'testTime', testTime, 'testResults',
                testResults, 'Test Pareto Acc', all_acc, 'Test Pareto Features', all_accTop,
                'Test Pareto Ensemble Acc', all_ensemble, 'Test Pareto Top Five Ensemble', all_topFiveE,
                'bestInd in training', hof[0], 'Best individual in pareto front', *hof[:],
                'final best fitness',best_ind_va[-1].fitness,'initial fitness', best_ind_va[0].fitness)

def saveAllResults_EnsembleMT(randomSeeds,dataSetName, best_ind_va,log,hof,num_features,trainTime,testTime,testResults, all_acc, all_accTop, all_ensemble, all_topFiveE):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime, 'trainResults',
                hof[0].fitness, 'number features', num_features, 'testTime', testTime, 'testResults',
                testResults, 'Test Pareto Acc', all_acc, 'Test Pareto Features', all_accTop,
                'Test Pareto Ensemble Acc', all_ensemble, 'Test Pareto Top Five Ensemble', all_topFiveE,
                'bestInd in training', hof[0][0],  hof[0][1], 'Best local in pareto front', *hof[:][0],'Best global in pareto front', *hof[:][1],
                'final best fitness',best_ind_va[-1].fitness,'initial fitness', best_ind_va[0].fitness)

def saveAllResults_mtflgp5(randomSeeds, dataSetName, log, best_tree, all_best, best_train, best_num_feature, trainTime,testTime, best_test, algorithm, test_acc, num_features):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime, 'trainResults',
                best_train, 'number features', best_num_feature, 'testTime', testTime, 'testResults',
                best_test, 'Test Each Classifier', test_acc, 'Test Each Features', num_features,
                'Test Pareto Ensemble Acc', test_acc[-1], 'Test Algorithm', algorithm,
                'bestInd in training', best_tree[0][0],  best_tree[0][1], 'Best local in pareto front', *all_best[:][0],'Best global in pareto front', *all_best[:][1])

def saveAllResults_Ensemble_mtmo(randomSeeds,dataSetName, best_ind_va,log,hof,num_features,trainTime,testTime,testResults, all_acc, all_accTop, all_ensemble, all_topFiveE):
    fileName1= str(randomSeeds)+'Resultson' + dataSetName+ '.txt'
    saveLog(fileName1, log)
    fileName2=str(randomSeeds)+'FinalResultson' + dataSetName+ '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime, 'trainResults',
                hof[0].fitness, 'number features', num_features, 'testTime', testTime, 'testResults',
                testResults, 'Test Pareto Acc', all_acc, 'Test Pareto Features', all_accTop,
                'Test Pareto Ensemble Acc', all_ensemble, 'Test Pareto Top Five Ensemble', all_topFiveE,
                'bestInd in training', hof[0][0],  hof[0][1],
                'final best fitness',best_ind_va[-1].fitness,'initial fitness', best_ind_va[0].fitness)
