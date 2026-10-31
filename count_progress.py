# -*- coding: utf-8 -*-
"""
进度统计脚本
统计三台电脑 (PC1 / PC2 / PC3) 各自的实验任务分别运行到第几个 seed。

用法:
    python count_progress.py [--root 结果根目录] [--target 每组目标 seed 数]

默认结果根目录: E:\\SLGP论文实验结果\\第一轮_大修
默认目标 seed 数: 30

目录结构约定 (结果收集后):
    {root}/PC{n}/{method}/{log|pkl|csv}/{config}/{dataset}_{seed}.{ext}
    例: E:\\SLGP论文实验结果\\第一轮_大修\\PC1\\FGP\\log\\CIFAR10-LT-400\\cifar10_1.log

统计规则:
  - 以 log/ 目录下每个 config 里的 *.log 文件数作为"已完成 seed 数"
  - 同时统计 pkl / csv 数量供核对
  - 若 seed 编号存在缺口 (例如跑到 1,2,4 缺 3), 会列出缺失的 seed
"""

import argparse
import os
import re
from collections import defaultdict

# 匹配 seed 文件名如 cifar10_3.log / MRD_12.pkl
SEED_RE = re.compile(r'_(\d+)\.(log|pkl|csv)$', re.IGNORECASE)


def count_config(config_dir):
    """统计某个 config 目录下的完成情况, 返回 (log数, pkl数, csv数, seed集合)."""
    log_n = pkl_n = csv_n = 0
    seeds = set()
    if not os.path.isdir(config_dir):
        return 0, 0, 0, set()
    for fn in os.listdir(config_dir):
        m = SEED_RE.search(fn)
        if m:
            seeds.add(int(m.group(1)))
        ext = m.group(2).lower() if m else os.path.splitext(fn)[1].lower()
        if ext == 'log':
            log_n += 1
        elif ext == 'pkl':
            pkl_n += 1
        elif ext == 'csv':
            csv_n += 1
    return log_n, pkl_n, csv_n, seeds


def missing_seeds(seeds):
    """若 1..max 之间存在缺口, 返回缺失 seed 列表 (最多显示 20 个)."""
    if not seeds:
        return []
    mx = max(seeds)
    missing = sorted(set(range(1, mx + 1)) - seeds)
    return missing[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'E:\SLGP论文实验结果\第一轮_大修',
                    help='结果根目录 (含 PC1/PC2/PC3 子目录)')
    ap.add_argument('--target', type=int, default=30,
                    help='每组实验目标 seed 数 (默认 30)')
    args = ap.parse_args()

    root = args.root
    target = args.target
    if not os.path.isdir(root):
        print('[ERROR] result root not found:', root)
        print('        usage: python count_progress.py --root <结果目录> --target 30')
        return

    # data[pc][method][config] = (log, pkl, csv, seeds)
    data = defaultdict(lambda: defaultdict(dict))
    for pc in sorted(os.listdir(root)):
        pc_path = os.path.join(root, pc)
        if not os.path.isdir(pc_path) or not pc.upper().startswith('PC'):
            continue
        for method in sorted(os.listdir(pc_path)):
            method_path = os.path.join(pc_path, method)
            if not os.path.isdir(method_path):
                continue
            log_dir = os.path.join(method_path, 'log')
            if not os.path.isdir(log_dir):
                continue
            for config in sorted(os.listdir(log_dir)):
                config_path = os.path.join(log_dir, config)
                if not os.path.isdir(config_path):
                    continue
                data[pc][method][config] = count_config(config_path)

    if not data:
        print('[WARN] no PC*/method/log/config structure found under root.')
        print('       please check --root and the directory layout.')
        return

    print('=' * 100)
    print('Progress: {done}/{target} per (PC, method, config)'.format(
        done='done', target='target'))
    print('=' * 100)

    for pc in sorted(data):
        print('\n### {pc} ###'.format(pc=pc))
        print('  {method:<16} {config:<30} {done:>4} / {target:<4} {note}'.format(
            method='Method', config='Config', done='done', target='target', note=''))
        pc_total = 0
        for method in sorted(data[pc]):
            for config in sorted(data[pc][method]):
                log_n, pkl_n, csv_n, seeds = data[pc][method][config]
                done = log_n
                pc_total += done
                miss = missing_seeds(seeds)
                note = ''
                if done < target:
                    note = '<-- INCOMPLETE'
                print('  {method:<16} {config:<30} {done:>4} / {target:<4} {note}'.format(
                    method=method, config=config, done=done, target=target, note=note))
                if miss:
                    print('      missing seeds: {m}'.format(m=', '.join(map(str, miss))))
        print('  {method:<16} {config:<30} {done:>4} / {target:<4}  (subtotal)'.format(
            method='[TOTAL]', config='', done=pc_total, target=len(data[pc]) * target))
    print('\n' + '=' * 100)
    print('Note: done = number of *.log files under log/<config>/')
    print('=' * 100)


if __name__ == '__main__':
    main()