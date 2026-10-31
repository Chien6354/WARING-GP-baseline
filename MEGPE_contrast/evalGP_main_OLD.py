import random
from deap import tools
import numpy as np
import selectBest
import copy
import pickle

def eaSimple(population, toolbox, cxpb, mutpb, ngen, start_gen ):
    if start_gen > ngen:
        return population
    pop_num=len(population)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    for gen in range(start_gen, ngen + 1):
        offspring = CrossoverAndVariation(population, cxpb, toolbox, mutpb, pop_num)
        population = Unique(offspring + population)
        while len(population) < pop_num:
            offspring = CrossoverAndVariation(population, cxpb, toolbox, mutpb, pop_num)
            population = Unique(offspring + population)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for idx,(ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            print(f"第{gen}代第{idx + 1}个个体")
            ind.fitness.values = fit
        population = toolbox.select(population, pop_num)
        checkpoint_data = {'population': population, 'generation': gen}
        with open("Fashion_mnist.pkl", "wb") as cp_file:
            pickle.dump(checkpoint_data, cp_file)
        print("去重后选择的150个体:{}".format(len(population)))
        for idx in population:
            print(idx)
    return population


import threading
import queue

def Unique(oldList):
    print("去重:{}".format(len(oldList)))
    unique_list = []
    for obj1 in oldList:
        for obj2 in unique_list:
            if obj1 == obj2:
                break
        else:
            unique_list.append(obj1)
    print("去重后的个体数:{}".format(len(unique_list)))
    return unique_list

def CrossoverAndVariation(population,cxpb,toolbox,mutpb, pop_num):
    new_cxpb = cxpb / (cxpb + mutpb)
    #添加拥挤度
    offspring=[]
    population = toolbox.select(population, len(population))
    for i in range(0, pop_num):
        #从原始种群中随机选5个
        aspirants = tools.selRandom(population, 5)
        #从五个中选择最好的一个
        offspring.append(copy.deepcopy(selectBest.selectBestFromAspirants(aspirants)))
    print("交叉变异时的个体数:{}".format(len(offspring)))
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring




