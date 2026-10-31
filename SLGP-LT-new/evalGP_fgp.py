import pickle
import random
from deap import tools, gp
from collections import defaultdict
from main import parallel_evalTrain, x_train, y_train


def pop_compare(ind1, ind2):
    from collections import defaultdict
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(idx)
    return types1 == types2

def varAndPaired(offspring_a, offspring_b, toolbox_a, toolbox_b, cxpb, mutpb, pset_1=None):
    """
    对两个种群同步进行交叉和变异，保持索引配对一致。

    行为：
      - 对每个相邻成对 (i, i+1)：
          * 若 random < cxpb：对 pair 执行交叉（mate）(a对a, b对b)
          * 否则：对 pair 的每个【配对位置】以概率 mutpb 执行变异
                  → (a[i], b[i]) 共享同一随机数：同时变异或同时不变
                  → (a[i+1], b[i+1]) 共享同一随机数：同时变异或同时不变
                  → 两个位置之间相互独立
      - 若长度为奇数，最后一个单独个体按 mutpb 决定是否变异（a/b同步）
      - 被修改的个体会使其 fitness 无效（del ind.fitness.values）
    """
    # 克隆，避免修改原始输入
    off_a = [toolbox_a.clone(ind) for ind in offspring_a]
    off_b = [toolbox_b.clone(ind) for ind in offspring_b]

    n = len(off_a)
    i = 0
    while i + 1 < n:
        a1, a2 = off_a[i], off_a[i+1]
        b1, b2 = off_b[i], off_b[i+1]

        if random.random() < cxpb:
            # 交叉：对两个种群同时交叉对应位置
            a1_new, a2_new = toolbox_a.mate(a1, a2)
            b1_new, b2_new = toolbox_b.mate(b1, b2)
            off_a[i], off_a[i+1] = a1_new, a2_new
            off_b[i], off_b[i+1] = b1_new, b2_new

            # 标记 fitness 无效
            try:
                del off_a[i].fitness.values
                del off_a[i+1].fitness.values
            except Exception:
                pass
            try:
                del off_b[i].fitness.values
                del off_b[i+1].fitness.values
            except Exception:
                pass

        else:
            # ── 位置 i：a[i] 与 b[i] 共享同一随机数，同步决定变异 ──
            if random.random() < mutpb:
                a_new, = toolbox_a.mutate(a1)
                off_a[i] = a_new
                try:
                    del off_a[i].fitness.values
                except Exception:
                    pass

                b_new, = toolbox_b.mutate(b1)
                off_b[i] = b_new
                try:
                    del off_b[i].fitness.values
                except Exception:
                    pass
            # else: a[i] 和 b[i] 同时保持不变

            # ── 位置 i+1：a[i+1] 与 b[i+1] 共享同一随机数，同步决定变异 ──
            if random.random() < mutpb:
                a_new2, = toolbox_a.mutate(a2)
                off_a[i+1] = a_new2
                try:
                    del off_a[i+1].fitness.values
                except Exception:
                    pass

                b_new2, = toolbox_b.mutate(b2)
                off_b[i+1] = b_new2
                try:
                    del off_b[i+1].fitness.values
                except Exception:
                    pass
            # else: a[i+1] 和 b[i+1] 同时保持不变

        i += 2

    # 若为奇数，处理最后一个未成对元素（a/b 同步）
    if n % 2 == 1:
        last = n - 1
        if random.random() < mutpb:
            new_last_a, = toolbox_a.mutate(off_a[last])
            off_a[last] = new_last_a
            try:
                del off_a[last].fitness.values
            except Exception:
                pass

            new_last_b, = toolbox_b.mutate(off_b[last])
            off_b[last] = new_last_b
            try:
                del off_b[last].fitness.values
            except Exception:
                pass

    return off_a, off_b


def eaSimple_1(population, population_1, toolbox, toolbox_1, cxpb, mutpb, elitpb, ngen, start_gen, pkl_file, pset_1,
               x_train, y_train, steepness):
    import os
    import pickle
    import math
    import numpy as np
    from deap import tools

    N = len(population)
    if N != len(population_1):
        raise ValueError("population and population_1 must have the same length")

    # 实时统计打印函数（不保存历史，只算当前）
    def print_stats(gen_no):
        vals = [ind.fitness.values[0] for ind in population if getattr(ind.fitness, "valid", False)]
        if len(vals) == 0:
            avg, mx = 0.0, 0.0
        else:
            avg = float(sum(vals)) / len(vals)
            mx = float(max(vals))
        print(f"Gen {gen_no}: avg fitness = {round(avg, 6)}, max fitness = {round(mx, 6)}")

    # 初始 evaluate 注册（如果从断点恢复，使用上一代作为 current_gen）
    initial_gen = start_gen - 1 if start_gen > 1 else 0
    toolbox.register("evaluate", parallel_evalTrain, current_gen=initial_gen, x_train=x_train, y_train=y_train,
                     ngen=ngen,
                     steepness=steepness)
    eval_pair_fn = toolbox.evaluate

    # 初始评估：对每个索引，如果任一侧 fitness 无效，则把该索引的两侧个体作为一对进行评估
    pairs_to_eval = []
    eval_positions = []
    for i in range(N):
        if (not getattr(population[i].fitness, "valid", False)) or (
        not getattr(population_1[i].fitness, "valid", False)):
            pairs_to_eval.append((population[i], population_1[i]))
            eval_positions.append(i)

    if pairs_to_eval:
        fitnesses = list(toolbox.map(eval_pair_fn, pairs_to_eval))
        for pos, fit in zip(eval_positions, fitnesses):
            fv = fit if isinstance(fit, tuple) else (fit,)
            population[pos].fitness.values = fv
            population_1[pos].fitness.values = fv

    # 如果是全新开始第二阶段，打印一次初始状态
    if start_gen == 1:
        print_stats(0)

    for gen in range(start_gen, ngen + 1):
        print(f"=== 第 {gen} 代 ===")
        # 重新注册评估函数以带入 current_gen
        toolbox.register("evaluate", parallel_evalTrain, current_gen=gen, x_train=x_train, y_train=y_train, ngen=ngen,
                         steepness=steepness)
        eval_pair_fn = toolbox.evaluate

        # 计算精英数量
        elitismNum = max(1, int(round(elitpb * N)))

        # id -> index 映射
        id2index_a = {id(ind): idx for idx, ind in enumerate(population)}

        # 选择精英
        elite_indices = []
        if elitismNum > 0:
            elites_a = tools.selBest(population, k=elitismNum)
            elite_indices = [id2index_a[id(ind)] for ind in elites_a]

        # 选择其余个体（非精英）
        non_elite_num = N - len(elite_indices)
        non_elite_indices = []
        selected_a = []
        selected_b = []
        if non_elite_num > 0:
            selected_individuals = toolbox.select(population, non_elite_num)
            non_elite_indices = [id2index_a[id(ind)] for ind in selected_individuals]
            selected_a = [toolbox.clone(population[i]) for i in non_elite_indices]
            selected_b = [toolbox.clone(population_1[i]) for i in non_elite_indices]

        # 配对交叉/变异
        varied_a, varied_b = varAndPaired(selected_a, selected_b, toolbox, toolbox_1, cxpb, mutpb, pset_1)

        # 构造位置级的评估队列
        to_eval_positions = []
        to_eval_pairs = []
        for pos, orig_idx in enumerate(non_elite_indices):
            ind_a = varied_a[pos]
            ind_b = varied_b[pos]
            if (not getattr(ind_a.fitness, "valid", False)) or (not getattr(ind_b.fitness, "valid", False)):
                to_eval_positions.append(pos)
                to_eval_pairs.append((ind_a, ind_b))

        # 并行评估
        if to_eval_pairs:
            fitnesses = list(toolbox.map(eval_pair_fn, to_eval_pairs))
            for pos, fit in zip(to_eval_positions, fitnesses):
                fv = fit if isinstance(fit, tuple) else (fit,)
                varied_a[pos].fitness.values = fv
                varied_b[pos].fitness.values = fv

                orig_idx = non_elite_indices[pos]
                if (not getattr(population[orig_idx].fitness, "valid", False)) or (
                        fv[0] > population[orig_idx].fitness.values[0]):
                    population[orig_idx].fitness.values = fv
                    population_1[orig_idx].fitness.values = fv

        # 构造新一代 offspring
        offspring = []
        offspring_1 = []

        for idx in elite_indices:
            a = toolbox.clone(population[idx])
            b = toolbox.clone(population_1[idx])
            if getattr(population[idx].fitness, "valid", False):
                a.fitness.values = population[idx].fitness.values
            if getattr(population_1[idx].fitness, "valid", False):
                b.fitness.values = population_1[idx].fitness.values
            offspring.append(a)
            offspring_1.append(b)

        offspring.extend(varied_a)
        offspring_1.extend(varied_b)

        if len(offspring) < N:
            used = set(elite_indices) | set(non_elite_indices)
            for i in range(N):
                if len(offspring) >= N:
                    break
                if i in used:
                    continue
                a = toolbox.clone(population[i])
                b = toolbox.clone(population_1[i])
                if getattr(population[i].fitness, "valid", False):
                    a.fitness.values = population[i].fitness.values
                if getattr(population_1[i].fitness, "valid", False):
                    b.fitness.values = population_1[i].fitness.values
                offspring.append(a)
                offspring_1.append(b)

        offspring = offspring[:N]
        offspring_1 = offspring_1[:N]

        population[:] = offspring
        population_1[:] = offspring_1

        # 打印当前代统计信息
        print_stats(gen)

        # 极简 Checkpoint 保存
        checkpoint_data = {
            'population': population,
            'population_1': population_1,
            'generation': gen,
        }
        try:
            with open(pkl_file, "wb") as cp_file:
                pickle.dump(checkpoint_data, cp_file)
        except Exception as e:
            print("Warning: failed to write checkpoint:", e)

    # 提取并返回最优个体
    best_a = tools.selBest(population, k=1)[0]
    best_b = tools.selBest(population_1, k=1)[0]

    return best_a, best_b


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if (offspring[i - 1] == offspring[i]) or pop_compare(offspring[i - 1], offspring[i]):
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring

def eaSimple(population, toolbox, cxpb, mutpb, elitpb, ngen, start_gen, pkl_file):

    import pickle

    # ====== 新增：记录每代 avg/max fitness ======
    gens = []
    avg_fitnesses = []
    max_fitnesses = []

    def compute_and_record_stats(gen_no):
        vals = [ind.fitness.values[0] for ind in population if getattr(ind.fitness, "valid", False)]
        if len(vals) == 0:
            avg = 0.0
            mx = 0.0
        else:
            avg = float(sum(vals)) / len(vals)
            mx = float(max(vals))
        avg_r = round(avg, 6)
        mx_r = round(mx, 6)
        print(f"Gen {gen_no}: avg fitness = {avg_r}, max fitness = {mx_r}")
        gens.append(gen_no)
        avg_fitnesses.append(avg_r)
        max_fitnesses.append(mx_r)
    # ==========================================

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    print("使用evaluate_0")
    fitnesses = toolbox.map(toolbox.evaluate_0, invalid_ind)

    # 原代码这里 zip(population, fitnesses) 有问题：会错配；按 invalid_ind 赋值更合理
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit



    hof_store = tools.HallOfFame(5 * len(population))
    hof_store.update(population)


    # ====== 新增：记录初始统计（start_gen - 1） ======
    compute_and_record_stats(start_gen - 1)
    # ==============================================

    for gen in range(start_gen, ngen + 1):
        # Select the next generation individuals by elitism
        elitismNum = int(elitpb * len(population))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)

        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        for i in offspring:
            ind = 0
            while ind < len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    ind = len(hof_store)
                else:
                    ind += 1

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate_0, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring[0:0] = offspringE

        cop_po = offspring.copy()
        hof_store.update(offspring)
        for i in hof_store:
            cop_po.append(i)

        population[:] = offspring

        # ====== 新增：记录当前代统计（注意 population 已更新） ======
        compute_and_record_stats(gen)
        # =======================================================
        checkpoint_data = {
            'population': population,
            'population_1': None,
            'generation': gen,
        }
        with open(pkl_file, "wb") as cp_file:
            pickle.dump(checkpoint_data, cp_file)

    return population
