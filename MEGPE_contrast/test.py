import pickle
import random

h=28
w=28
def make_Singma():
    return random.randint(1, 4)

def make_Order():
    return random.randint(0, 3)

def make_Theta():
    return random.randint(0, 8)

def make_Frequency():
    return random.randint(0, 8)

def make_classificationN():
    return random.randint(1, 4)

def make_coordsX():
    return random.randint(0, h-3)

def make_coordsY():
    return random.randint(0, w-3)

def make_windowSize():
    return random.randint(3, w)

def make_width():
    return random.randint(3, w)

def make_height():
    return random.randint(3, h)

def make_reuse():
    return random.randint(0, 1)

# 从.pkl文件加载对象
with open('GPFR_MBI_False_1.pkl', 'rb') as f:
    checkpoint_data = pickle.load(f)
    pop = checkpoint_data['population']
    for i in pop:
        print(i)
    start_gen = checkpoint_data['generation'] + 1
print(start_gen)