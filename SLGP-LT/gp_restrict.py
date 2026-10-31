import random
import sys
import warnings
from inspect import isclass

# Define the name of type for any types.
__type__ = object

from deap import gp

from strongGPDataType import Img0


######################################
# GP Program generation functions    #
######################################
def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height
    #print('it works', pset)
    return generate(pset, min_, max_, condition, type_)

def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or depth >= min_
    return generate(pset, min_, max_, condition, type_)

def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    #print(method)
    return method(pset, min_, max_, type_)

def genRamped(pset, min_, max_, type_=None):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)

def generate(pset, min_, max_, condition, type_=__type__):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
    :param pset: A primitive set from wich to select primitives of the trees.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.


    DUMMY NODE ISSUES

    DEAP will only place terminals if we're at the bottom of a branch.
    This creates two issues:
    1. A primitive that takes other primitives as inputs could be placed at the
        second to last layer.
        SOLUTION: You need to allow the tree to end whenever the height condition is met,
                    so create "dummy" terminals for every type possible in the tree.
    2. A primitive that takes terminals as inputs could be placed above the second to
        last layer.
        SOLUTION: You need to allow the tree to continue extending the branch until the
                    height condition is met, so create "dummy" primitives that just pass
                    through the terminal types.

    These "dummy" terminals and "dummy" primitives introduce unnecessary and sometimes
    nonsensical solutions into populations. These "dummy" nodes can be eliminated
    if the height requirement is relaxed.


    HOW TO PREVENT DUMMY NODE ISSUES

    Relaxing the height requirement:
    When at the bottom of the branch, check for terminals first, then primitives.
        When checking for primitives, skirt the height requirement by adjusting
        the branch depth to be the second to last layer of the tree.
        If neither a terminal or primitive fits this node, then throw an error.
    When not at the bottom of the branch, check for primitives first, then terminals.

    Issue with relaxing the height requirement:
    1. Endless loops are possible when primitive sets have any type loops.
        A primitive with an output of one type may not take an input type of
        itself or a parent type.
        SOLUTION: A primitive set must be well-designed to prevent those type loops.

    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    #print(len(stack))
    #print(pset.terminals)
    #print(pset.primitives)
    while len(stack) != 0:
##        print(len(expr))
        if (len(expr)>60):
            expr = []
            #if type_ is None:
            type_ = pset.ret
            stack = [(0, type_)]
##            print(depth, type_)
            #depth, type_ = stack.pop()
            height = random.randint(min_, max_)
        depth, type_ = stack.pop()
        if condition(height, depth):
            # Try finding a terminal
            try:
                term = random.choice(pset.terminals[type_])
                #print('term',term)
                if isclass(term):
                    term = term()
                expr.append(term)
                # No terminal fits
            except:
                # So pull the depth back one layer, and start looking for primitives
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
                    #print('prim',prim)
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth, arg))

                        # No primitive fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a primitive of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
        # Not at the bottom of the tree
        else:
            # Check for primitives
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
                    # No primitive fits
            except:
                # So check for terminals
                try:
                    term = random.choice(pset.terminals[type_])

                # No terminal fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a terminal of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
                if isclass(term):
                    term = term()
                expr.append(term)
    #print(len(expr))
    return expr


def generateMD(pset, min_, max_, condition, type_=__type__):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
##    print(len(stack))
    #print(pset.terminals)
    #print(pset.primitives)
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            # Try finding a terminal
            try:
                term = random.choice(pset.terminals[type_])
                #print('term',term)
                if isclass(term):
                    term = term()
                expr.append(term)
                # No terminal fits
            except:
                # So pull the depth back one layer, and start looking for primitives
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
                    #print('prim',prim)
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth, arg))

                        # No primitive fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a primitive of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
        # Not at the bottom of the tree
        else:
            # Check for primitives
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
                    # No primitive fits
            except:
                # So check for terminals
                try:
                    term = random.choice(pset.terminals[type_])

                # No terminal fits, either - that's an error
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add " \
                                      "a terminal of type '%s', but there is " \
                                      "none available." % (type_,), traceback)
                if isclass(term):
                    term = term()
                expr.append(term)
    #print(len(expr))
    return expr

def genHalfAndHalfMD(pset, min_, max_, type_=None):
    expr=genHalfAndHalf(pset, min_, max_, type_=None)
    #print(expr)
    #print('expr before', len(expr))
    while len(expr)>60:
        expr=genHalfAndHalf(pset, min_, max_, type_=None)
        #print('expr before', len(expr))
    #print('expr after',len(expr))
    return expr




def genFullMD(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) no return type is enforced.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height
    #print('it works', pset)]
    expr=generateMD(pset, min_, max_, condition, type_)
##    print(len(expr))
##    while len(expr)>20:
##        expr=generateMD(pset, min_, max_, condition, type_)
    return expr


def genConstrained(pset, min_, max_):
    height = random.randint(min_, max_)
    expr = []
    stack = [(0, pset.ret)]

    while stack:
        depth, type_ = stack.pop()

        # 如果已经达到最大深度，必须用 terminal
        if depth >= height:
            term = random.choice(pset.terminals[type_])
            if callable(term) and not isinstance(term, (int, float, str, bool)):
                term = term()
            expr.append(term)
            continue

        # 否则，尝试用 primitive（如果存在）
        primitives = pset.primitives.get(type_, [])
        if primitives:
            prim = random.choice(primitives)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
        else:
            # 没有 primitive？那就用 terminal（提前终止该分支）
            term = random.choice(pset.terminals[type_])
            if callable(term) and not isinstance(term, (int, float, str, bool)):
                term = term()
            expr.append(term)
            # 注意：该分支提前结束，树高度可能 < height（但整体 >= min_ 由调用方保证）

    return expr

def compute_depths(tree):
    """返回每个节点的深度列表"""
    depths = [0] * len(tree)
    stack = [0]
    while stack:
        i = stack.pop()
        depth = depths[i]
        node = tree[i]
        if isinstance(node, gp.Primitive):
            for j in range(i + 1, i + 1 + node.arity):
                if j < len(tree):
                    depths[j] = depth + 1
                    stack.append(j)
    return depths

def subtree_slice(tree, idx):
    """返回从 idx 开始的子树切片长度"""
    node = tree[idx]
    if isinstance(node, gp.Terminal):
        return 1
    else:
        size = 1
        for _ in range(node.arity):
            if idx + size < len(tree):
                child_size = subtree_slice(tree, idx + size)
                size += child_size
            else:
                break
        return size

def tree_height_from_index(tree, idx):
    """计算从 idx 开始的子树高度"""
    node = tree[idx]
    if isinstance(node, gp.Terminal):
        return 0
    else:
        max_child_h = 0
        pos = idx + 1
        for _ in range(node.arity):
            if pos >= len(tree):
                break
            child_h = tree_height_from_index(tree, pos)
            max_child_h = max(max_child_h, child_h)
            pos += subtree_slice(tree, pos)
        return 1 + max_child_h


def cxConstrainedHeight(tree1, tree2, min_h=2, max_h=5, max_trials=20):
    """
    受高度约束的交叉。
    - 根节点（index 0）不可被选。
    - 保证交叉后两棵树高度 ∈ [min_h, max_h]
    """
    # 备份原始高度
    orig_h1 = tree_height_from_index(tree1, 0)
    orig_h2 = tree_height_from_index(tree2, 0)

    # 预计算深度
    depths1 = compute_depths(tree1)
    depths2 = compute_depths(tree2)

    # 所有可能的非根节点索引（跳过根）
    candidates1 = [i for i in range(1, len(tree1))]
    random.shuffle(candidates1)

    for trial in range(max_trials):
        if not candidates1:
            break

        idx1 = candidates1.pop()
        node1 = tree1[idx1]

        # 跳过终端（虽然可以交叉，但通常无意义；可保留）
        # 获取子树信息
        slice1 = subtree_slice(tree1, idx1)
        h1_sub = tree_height_from_index(tree1, idx1)
        depth1 = depths1[idx1]

        # 在 tree2 中找合法的 idx2
        valid_idx2_list = []
        for idx2 in range(1, len(tree2)):  # 不动根
            node2 = tree2[idx2]
            if node2.ret != node1.ret:
                continue

            slice2 = subtree_slice(tree2, idx2)
            h2_sub = tree_height_from_index(tree2, idx2)
            depth2 = depths2[idx2]

            # 检查新高度是否合法
            new_h1 = orig_h1 - h1_sub + h2_sub
            new_h2 = orig_h2 - h2_sub + h1_sub

            if min_h <= new_h1 <= max_h and min_h <= new_h2 <= max_h:
                valid_idx2_list.append(idx2)

        if valid_idx2_list:
            idx2 = random.choice(valid_idx2_list)

            # 执行交叉
            slice1_len = subtree_slice(tree1, idx1)
            slice2_len = subtree_slice(tree2, idx2)

            new_tree1 = gp.PrimitiveTree(
                tree1[:idx1] + tree2[idx2:idx2 + slice2_len] + tree1[idx1 + slice1_len:]
            )
            new_tree2 = gp.PrimitiveTree(
                tree2[:idx2] + tree1[idx1:idx1 + slice1_len] + tree2[idx2 + slice2_len:]
            )

            return new_tree1, new_tree2

    # 如果所有尝试失败，返回原树（或做安全回退）
    return tree1, tree2

import random
from deap import gp

# def mutUniformHeight(individual, pset, min_h=2, max_h=5):
#     """
#     Uniform mutation with height constraint [min_h, max_h] for the whole tree.
#     - Root (index 0) is NOT mutated (must remain 'Change').
#     - New subtree height is limited so that total height <= max_h.
#     """
#     if len(individual) <= 1:
#         return individual,
#
#     # Step 1: 随机选择一个非根节点
#     idx = random.randint(1, len(individual) - 1)  # skip root (index 0)
#
#     # Step 2: 计算当前整树高度和节点深度
#     def get_depths_and_height(tree):
#         depths = [0] * len(tree)
#         stack = [0]
#         while stack:
#             i = stack.pop()
#             node = tree[i]
#             if isinstance(node, gp.Primitive):
#                 for j in range(i + 1, i + 1 + node.arity):
#                     if j < len(tree):
#                         depths[j] = depths[i] + 1
#                         stack.append(j)
#         tree_height = max(depths) if depths else 0
#         return depths, tree_height
#
#     depths, current_total_h = get_depths_and_height(individual)
#     d = depths[idx]  # 当前节点深度
#     type_ = individual[idx].ret  # 子树返回类型
#
#     # Step 3: 计算允许的新子树最大高度
#     max_sub_h = max_h - d  # 从深度 d 开始，子树最多高 (max_h - d)
#     if max_sub_h < 0:
#         max_sub_h = 0
#     if max_sub_h == 0:
#         # 只能插入 terminal
#         if pset.terminals[type_]:
#             new_sub = [random.choice(pset.terminals[type_])]
#         else:
#             return individual,  # 无法变异
#     else:
#         # 生成高度不超过 max_sub_h 的子树
#         # 使用 gp.genGrow（更灵活）或 genFull
#         min_gen = 0
#         max_gen = max_sub_h
#         # 但至少生成一个节点
#         if max_gen == 0:
#             max_gen = 1
#         new_sub = None
#         for _ in range(10):
#             try:
#                 new_sub = gp.genGrow(pset, min_=min_gen, max_=max_gen, type_=type_)
#                 # 检查生成的子树高度是否真的 <= max_sub_h
#                 _, h_new = get_depths_and_height(new_sub)
#                 if h_new <= max_sub_h:
#                     break
#             except Exception:
#                 continue
#         if new_sub is None:
#             # fallback: use a terminal if possible
#             if pset.terminals[type_]:
#                 new_sub = [random.choice(pset.terminals[type_])]
#             else:
#                 return individual,
#
#     # Step 4: 替换子树
#     slice_ = individual.searchSubtree(idx)
#     individual[slice_] = new_sub
#
#     # Step 5: 检查最终高度是否合法（兜底）
#     _, final_h = get_depths_and_height(individual)
#     if final_h < min_h or final_h > max_h:
#         # 如果不合法，撤销变异（或接受？这里选择撤销）
#         # 但通常不会发生，因为 max_sub_h 已限制
#         # 为安全起见，可回滚，但实现复杂；此处直接返回（或给低 fitness）
#         pass
#
#     return individual,

import random
from deap import gp

def mutUniformHeight(individual, pset, min_h=2, max_h=5):
    """
    简化版的带高度约束的变异：
    - 随机选非根节点作为变异点（若想包含根，把 randint 起点改为 0）。
    - 计算允许的子树高度范围 [min_sub_h, max_sub_h]（相对于变异点）。
    - 随机选一个目标子树高度 target_h，然后生成该高度约束下的完整子树并替换。
    返回 (individual,).
    """
    if len(individual) <= 1:
        return individual,
    assert hasattr(individual, 'fitness'), (
        f"mutUniformHeight received raw PrimitiveTree (type={type(individual)}). "
        "This means your population contains invalid individuals!"
    )
    # 计算每个节点深度与整树高度
    def get_depths_and_height(tree):
        depths = [0] * len(tree)
        stack = [0]
        while stack:
            i = stack.pop()
            node = tree[i]
            # 对于 primitive，推进其参数位置
            for j in range(i + 1, i + 1 + node.arity):
                if j < len(tree):
                    depths[j] = depths[i] + 1
                    stack.append(j)
        tree_height = max(depths) if depths else 0
        return depths, tree_height

    depths, curr_total_h = get_depths_and_height(individual)

    # 随机选变异点（跳过根节点 index 0）
    idx = random.randint(1, len(individual) - 1)

    node_depth = depths[idx]
    type_ = individual[idx].ret

    # 计算子树高度允许范围（相对于变异点）
    min_sub_h = max(0, min_h - node_depth)
    max_sub_h = max_h - node_depth
    if max_sub_h < 0:
        # 已经超过允许高度，无法变异
        return individual,

    if min_sub_h > max_sub_h:
        # 无合法高度范围
        return individual,

    target_h = random.randint(min_sub_h, max_sub_h)

    # 生成满足 target_h 的子树节点列表（前缀表示法）
    # 递归构造：当 curr_depth == target_h 时必须放 terminal；否则优先放 primitive（若无则放 terminal）
    def gen_subtree_nodes(type_req, curr_depth, target_h):
        # 到达目标深度：必须选择 terminal
        terminals = pset.terminals.get(type_req, [])
        primitives = pset.primitives.get(type_req, [])

        if curr_depth >= target_h:
            # 必须选择 terminal（用户保证类型上有 terminal）
            if not terminals:
                raise ValueError(f"No terminal available for type {type_req}")
            term = random.choice(terminals)
            return [term]

        # 未到达目标深度，优先选择 primitive（若无 primitive 则选择 terminal）
        if primitives:
            prim = random.choice(primitives)
            nodes = [prim]
            # 对每个参数递归生成对应的子树
            # prim.args 提供参数类型列表（DEAP Primitive 有 .args 属性）
            for arg_type in prim.args:
                child_nodes = gen_subtree_nodes(arg_type, curr_depth + 1, target_h)
                nodes.extend(child_nodes)
            return nodes
        else:
            # 没有 primitive 可用，退化为 terminal（作为分支终端）
            if not terminals:
                raise ValueError(f"No terminal available for type {type_req}")
            term = random.choice(terminals)
            return [term]

    # 生成新子树并替换
    try:
        new_nodes = gen_subtree_nodes(type_, 0, target_h)
        new_sub = gp.PrimitiveTree(new_nodes)
    except Exception:
        # 生成失败，回退不变
        return individual,

    # 替换子树 slice
    try:
        slice_ = individual.searchSubtree(idx)
        individual[slice_] = new_sub
    except Exception:
        # 若替换失败（例如不完整子树），不变更原个体
        return individual,

    # 可选：最终检查高度是否在范围内（作为保险）
    _, final_h = get_depths_and_height(individual)
    if final_h < min_h or final_h > max_h:
        # 回滚为原样（简单返回不变）
        return individual,

    return individual,
