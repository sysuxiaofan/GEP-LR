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
import pandas as pd
syntheticData = pd.read_excel('F:/GEP-LR/synthetic-data.xlsx')
x1 = syntheticData.x1.values   # x1 and x2 are variables
x2 = syntheticData.x2.values
Y = syntheticData.y1.values    # this is our target, now mapped to Y
#Creating the primitives set
import operator
#define a protected division to avoid dividing by zero
def protected_div(x, y):  
    if y==0.0:
        return np.nan
    return operator.truediv(x, y)
#
#Map our input data to the GEP variables
import geppy as gep
pset = gep.PrimitiveSet('Main', input_names=['x1','x2'])
#
#Define the operators
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_rnc_terminal()
#
#Create the individual and population
from deap import creator, base, tools
#to maximize the objective (fitness)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

#Register the individual and population creation operations
h = 10           # head length
n_genes = 1      # number of genes in a chromosome
r = 5            # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique
toolbox = gep.Toolbox()
#each RNC is random integer within [-10, 10]
toolbox.register('rnc_gen', random.uniform, a=-10, b=10)   
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, 
                 rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, 
                 gene_gen=toolbox.gene_gen, 
                 n_genes=n_genes, linker = None)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#'compile' translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

#Define logistic function
def logistic(x):
    x=x*1.0
    x[x>=0]=1.0/(1+np.exp(-x[x>=0]))
    x[x< 0]=np.exp(x[x<0])/(1+np.exp(x[x<0]))
    return x
#
from sklearn.metrics import roc_curve, auc
from numba import jit
#Calculate and accelerate the fitness function
@jit
def evaluate(individual):
    """Evalute the fitness of an individual: AUC (area under ROC)"""
    func = toolbox.compile(individual)
    ind_vals = np.array(list(map(func, x1, x2)))
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
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.10)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.10)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.10)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.10)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.10)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.10)
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
#Size of population and number of generations
n_pop = 100     #size of population
n_gen = 999     #number of generations, start from 0
champs = 3      #number of best individuals
#
#initial population 
pop = toolbox.population(n=n_pop) 
#only record the best individuals ever found in all generations
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
# print the best symbolic regression we found:
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)
from sympy import init_printing
init_printing()
#use str(symplified_best) to get the string of the symplified model
symplified_best 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Visualization
#we want to use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}
gep.export_expression_tree(best_ind, rename_labels, 
                           'F:/GEP-LR/synthetic-data-modeling-tree.png')
#show the above image here for convenience
from IPython.display import Image
Image(filename='F:/GEP-LR/synthetic-data-modeling-tree.png')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot maximum fitness values
import matplotlib.pyplot as plt
max_Fitness_values = log.select("max")
fig =  plt.figure(figsize=(15,5))
plt.plot(max_Fitness_values,'-bo')          # predictions are in red
plt.show()
fig.savefig('F:/GEP-LR/synthetic-data-modeling-maxFitness.eps', 
            dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Calculate the final estimated probability
best_func = toolbox.compile(best_ind)
predY = logistic(np.array(list(map(best_func, syntheticData.x1, 
                                     syntheticData.x2))))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot ROC of the result
classfied_predY = predY.copy()
classfied_predY[classfied_predY> 0.5] = 1
classfied_predY[classfied_predY<=0.5] = 0
fpr,tpr,threshold = roc_curve(syntheticData.y1, classfied_predY, 
                              pos_label=1, drop_intermediate=False)
roc_auc = auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.figure()
lw = 2
fig = plt.figure(figsize=(10,10))
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
fig.savefig('F:/GEP-LR/synthetic-data-modeling-ROC.eps', 
            dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot scatters
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,15))
plt.scatter(syntheticData.x1, syntheticData.x2, c=classfied_predY,
            marker='s')
plt.show()
fig.savefig('F:/GEP-LR/synthetic-data-modeling-scatters.eps', 
            dpi=300, format='eps')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Save GEP-LR results
#save maximum fitness function values
writer = pd.ExcelWriter('F:/GEP-LR/synthetic-data-modeling-maxFitness.xlsx')
data1 = {'max_Fitness': max_Fitness_values}
df1 = pd.DataFrame(data=data1)
df1.to_excel(writer,sheet_name='max_Fitness')
writer.save()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#save prediction results
writer = pd.ExcelWriter('F:/GEP-LR/synthetic-data-modeling-results.xlsx')
data2={'probability': predY,'clssification': classfied_predY}
df2 = pd.DataFrame(data=data2)
df2.to_excel(writer, sheet_name='results')
writer.save()