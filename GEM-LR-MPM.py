# -*- coding: utf-8 -*-
"""
Python 3.7
Created on Wed Apr 22 18:47:50 2020
@author: XiaoFan
"""
#for reproduction
import numpy as np
import random
s = 0
random.seed(s)
np.random.seed(s)

#import dataset
import pandas as pd
dtsMPMdata = pd.read_excel('F:/GEP-LR/dts_evidences.xls')
# A,B,C,D,and E are variables
A = dtsMPMdata.A.values
B = dtsMPMdata.B.values
C = dtsMPMdata.C.values
D = dtsMPMdata.D.values
E = dtsMPMdata.E.values
Y = dtsMPMdata.ORE.values  # this is our target, now mapped to Y

#Creating the primitives set
import operator
import math
#define a protected division to avoid dividing by zero
def protected_div(x, y):  
    if y==0.0:
        return np.nan
    return operator.truediv(x, y)
#define a protected logrithm to avoid log-transformed by zero
def protected_log(x):  
    if x <= 0.0:
        return np.nan
    return math.log(x)
#define a protected pow to avoid dividing by zero
def protected_pow(x, y): 
    if x==0.0:
        return 0.0
    else:
        if y<0.0:
            x=1.0/x
            y=abs(y)
    if x>0.0:
        if y*math.log(x)>=32.0:
            return 1.0e16
    if  x<0.0:
        if (y%2==0) & (y*math.log(-x) >= 32.0):
            return 1.0e16
        if (y%2==1) & (y*math.log(-x) >= 32.0): 
            return -1.0e16
        if (y%2!=0)&(y%2!=1):
           return np.nan
    return math.pow(x, y)
#define a protected exp to avoid dividing by zero    
def protected_exp(x):  
    if x >= 16.0:
        return 1.0e16
    return math.exp(x)
#define a protected sqrt to avoid dividing by zero
def protected_sqrt(x):  
    if x <= 0.0:
        return np.nan
    return math.sqrt(x)
#define a protected division to avoid dividing by zero
def protected_asin(x): 
    if abs(x) > 1.0:
        return np.nan
    return math.asin(x)

# Map our input data to the GEP variables
import geppy as gep
pset = gep.PrimitiveSet('Main', input_names=['A','B','C','D','E'])

#Define the operators
#F1
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
#F2
pset.add_function(protected_pow, 2)
pset.add_function(protected_log, 1)
pset.add_function(protected_exp, 1)
pset.add_function(protected_sqrt, 1)
#F3
pset.add_function(math.sin, 1)
pset.add_function(math.cos, 1)
pset.add_function(protected_asin, 1)
pset.add_function(math.atan, 1)
#Terminal of constants
pset.add_rnc_terminal()

#Create the individual and population
from deap import creator, base, tools
#to maximize the objective (fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

#Register the individual and population creation operations
h = 15           # head length
n_genes = 3      # number of genes in a chromosome
r = 10           # length of the RNC array
#
toolbox = gep.Toolbox()
#each RNC is random integer within [-10, 10]
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, 
                 rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
#Define link function
def linked_add(a, b, c):
    return operator.add(operator.add(a, b), c)
toolbox.register('individual', creator.Individual, 
                 gene_gen=toolbox.gene_gen, 
                 n_genes=n_genes, linker=linked_add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#'compile' translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

#Define logistic function
def logistic(x):
    x=x*1.0
    x[x>=0]=1.0/(1+np.exp(-x[x>=0]))
    x[x< 0]=np.exp(x[x<0])/(1+np.exp(x[x<0]))
    return x

from sklearn.metrics import roc_curve, auc
from numba import jit
#Calculate and accelerate the fitness function
@jit
def evaluate(individual):
    """Evalute the fitness of an individual: AUC (area under ROC)"""
    func = toolbox.compile(individual)
    ind_vals = np.array(list(map(func, A, B, C, D, E)))
    if np.isnan(ind_vals).any():
        return -1.0,
    else:
        Yp =logistic(ind_vals)  # probability of one
        # return AUC
        Yp[Yp> 0.5]=1
        Yp[Yp<=0.5]=0
        fpr,tpr,threshold = roc_curve(Y, Yp)
        return auc(fpr,tpr),
#
toolbox.register('evaluate', evaluate)
#
#Register genetic operators
toolbox.register('select', tools.selTournament, tournsize=3)
#1.general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, 
                 ind_pb=0.10, pb=1.00)
toolbox.register('mut_invert', gep.invert, pb=0.05)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.05)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.05)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.05)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.03)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.03)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.03)
#2.Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.10, pb=1.00)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.05)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.05)
#for some uniform mutations, we can also assign the ind_pb a string 
#to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, 
                 rnc_gen=toolbox.rnc_gen, ind_pb='0.5p', pb=1.00)
#
#Statistics to be inspected
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
#
#Launch evolution
# size of population and number of generations
n_pop = 500     #size of population
n_gen = 1999    #number of generations, start from 0
champs = 3      #number of best individuals
#
pop = toolbox.population(n=n_pop) # 
#only record the best three individuals ever found in all generations
hof = tools.HallOfFame(champs)  
#
import datetime
startDT = datetime.datetime.now()
print (str(startDT))
#Start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=2,
                          stats=stats, hall_of_fame=hof, verbose=True)
print ("Evolution times were:\n\nStarted:\t", 
       startDT, "\nEnded:   \t", str(datetime.datetime.now()))
print(hof[0])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Symbolic simplification of the final solution
#print the best symbolic regression we found:
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)
from sympy import init_printing
init_printing()
#use str(symplified_best) to get the string of the symplified model
symplified_best 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Visualization
#use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}
gep.export_expression_tree(best_ind, rename_labels, 'F:/GEP-LR/MPM-tree.png')
#show the above image here for convenience
from IPython.display import Image
Image(filename='F:/GEP-LR/MPM-tree.png')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot maximum fitness values
import matplotlib.pyplot as plt
max_Fitness_values = log.select("max")
fig =  plt.figure(figsize=(15, 5))
plt.plot(max_Fitness_values,'-bo')     # predictions are in red
plt.show()
fig.savefig('F:/GEP-LR/MPM-maxFitness.eps', dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Calculate the final estimated probability
best_func = toolbox.compile(best_ind)
predORE = logistic(np.array(list(map(best_func, dtsMPMdata.A, 
                                     dtsMPMdata.B, dtsMPMdata.C, 
                                     dtsMPMdata.D, dtsMPMdata.E))))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot ROC of the result
classfied_predORE = predORE.copy()
classfied_predORE[classfied_predORE> 0.5] = 1
classfied_predORE[classfied_predORE<=0.5] = 0
fpr,tpr,threshold = roc_curve(dtsMPMdata.ORE, classfied_predORE, 
                              pos_label=1, drop_intermediate=False)
roc_auc = auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.figure()
lw = 2
fig = plt.figure(figsize=(10,10))
#假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr, tpr, color='darkorange', 
         lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic cure plot for GEP-LR')
plt.legend(loc="lower right")
plt.show()
fig.savefig('F:/GEP-LR/MPM-ROC.eps', dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot scatters
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
plt.scatter(dtsMPMdata.XX, dtsMPMdata.YY, c=classfied_predORE, marker='s')
plt.show()
fig.savefig('F:/GEP-LR/MPM-scatters.eps', dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Save results
#save maximum fitness function values
writer = pd.ExcelWriter('F:/GEP-LR/MPM-maxFitness.xlsx')
data1 = {'max_Fitness': max_Fitness_values}
df1 = pd.DataFrame(data=data1)
df1.to_excel(writer,sheet_name = 'max_Fitness')
writer.save()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#save prediction results
writer = pd.ExcelWriter('F:/GEP-LR/MPM-results.xlsx')
data2={'probability': predORE, 'classification': classfied_predORE}
df2 = pd.DataFrame(data=data2)
df2.to_excel(writer, sheet_name = 'results')
writer.save()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%