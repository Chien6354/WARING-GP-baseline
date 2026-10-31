import random


def selectBestFromAspirants(aspirants):
    """
    从候选人中选择最佳个体：
    1. 首先根据rank进行排序。
    2. 如果有多个个体具有相同的rank，则根据拥挤距离（crowding distance）进行排序。
    3. 如果仍有多个个体具有相同的rank和crowding distance，则选择树深度最小的个体。
    4. 如果树深度也相同，则随机选择一个个体。
    """

    # 按rank升序，crowding distance降序排序
    aspirants.sort(key=lambda ind: (ind.fitness.rank, -ind.fitness.crowding_dist))

    # 获取rank最小的个体
    best_rank = aspirants[0].fitness.rank
    candidates = [ind for ind in aspirants if ind.fitness.rank == best_rank]

    # 如果只有一个个体具有最佳rank，直接返回
    if len(candidates) == 1:
        return candidates[0]

    # 根据拥挤距离进一步筛选
    candidates.sort(key=lambda ind: -ind.fitness.crowding_dist)
    best_crowding_dist = candidates[0].fitness.crowding_dist
    candidates = [ind for ind in candidates if ind.fitness.crowding_dist == best_crowding_dist]

    # 如果只有一个个体具有最佳rank和crowding distance，直接返回
    if len(candidates) == 1:
        return candidates[0]

    # 根据树深度进一步筛选
    candidates.sort(key=lambda ind: len(ind))
    smallest_gp_tree = candidates[0]

    # 如果只有一个个体具有最小的树深度，直接返回
    if len([ind for ind in candidates if len(ind) == len(smallest_gp_tree)]) == 1:
        return smallest_gp_tree

    # 如果树深度也相同，则随机选择一个个体
    return random.choice(candidates)