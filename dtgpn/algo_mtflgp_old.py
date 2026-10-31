import pickle
import random
from deap import tools,creator,base
from operator import attrgetter
from collections import defaultdict
import logging
from HOF import HallOfFame

creator.create("CFitnessMax", base.Fitness, weights=(1.0,1.0))
creator.create("CIndividual", list, fitness=creator.CFitnessMax)

def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:],1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:],1):
        types2[node.ret].append(idx)
    return types1==types2

def varAnd(toolbox, cxpb, mutpb, pop_main):
    # print(pop_main)
    offsp= toolbox.clone(pop_main)
    offspring = [toolbox.clone(ind) for ind in offsp]
    new_cxpb = cxpb/(cxpb+mutpb)
    i = 0
    n= len(offspring)
    # print(len(offspring))
    while i < n-1:
        if random.random() < new_cxpb:
            # print(offspring[i].fitness.values, offspring[i+1].fitness.values)
            if (offspring[i][0] == offspring[i + 1][0]) or (pop_compare(offspring[i][0], offspring[i + 1][0])):
                # print('left mutation')
                offspring[i][0], = toolbox.mutate(offspring[i][0])
                offspring[i + 1][0], = toolbox.mutate(offspring[i + 1][0])
            else:
                # print('left crossover')
                offspring[i][0], offspring[i + 1][0] = toolbox.mate(offspring[i][0], offspring[i + 1][0])
            if (offspring[i][1] == offspring[i + 1][1]) or (pop_compare(offspring[i][1], offspring[i + 1][1])):
                # print('right mutation')
                offspring[i][1], = toolbox.mutate(offspring[i][1])
                offspring[i + 1][1], = toolbox.mutate(offspring[i + 1][1])
            else:
                # print('right crossover')
                offspring[i][1], offspring[i + 1][1] = toolbox.mate(offspring[i][1], offspring[i + 1][1])
            del offspring[i].fitness.values, offspring[i + 1].fitness.values
            i = i + 2
            # print('after crossover', offspring[i].fitness.values, offspring[i+1].fitness.values)
        else:
            # print('all mutation')
            offspring[i][0], = toolbox.mutate(offspring[i][0])
            offspring[i][1], = toolbox.mutate(offspring[i][1])
            del offspring[i].fitness.values
            i = i + 1
        # print('after', i)
    return offspring

def record_pop(pop, logbook, stats, gen):
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=gen, nevals=len(pop), **record)
    logging.info(logbook.stream)
    return logbook

def evalSimple( global_pop, local_pop1, toolbox, cxpb, mutpb,elitpb,  ngen , start_gen, pkl_file ):
    population = []
    print(len(population))
    print(ngen)
    for ind in range(len(local_pop1)):
        individual = creator.CIndividual([local_pop1[ind], global_pop[ind]])
        population.append(individual)

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    logging.info(len(population))
    hof = HallOfFame(10)
    hof.update(population)
    # for i in population:
        # print(i.fitness.values)
    for gen in range(start_gen, ngen + 1):
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        offspring = toolbox.select(population, len(population) - elitismNum)
        # print('select')
        # for i in offspring:
        #     print(i.fitness.values)
        offspring = varAnd(toolbox, cxpb, mutpb, pop_main = offspring)
        # print('a/fter')
        # for i in offspring:
            # print(i.fitness.values)
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        population= offspring+offspringE

        hof.update(offspring)

        checkpoint_data = {
            'population': population,
            'generation': gen,
        }
        try:
            with open(pkl_file, "wb") as cp_file:
                pickle.dump(checkpoint_data, cp_file)
        except Exception as e:
            print("Warning: failed to write checkpoint:", e)
    return hof

def selTournamentDCD(individuals, k):
    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))
    chosen = []
    for i in range(0, k, 4):
        chosen.append(tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn(individuals_2[i + 2], individuals_2[i + 3]))
    return chosen
