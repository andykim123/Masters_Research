#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 05:47:27 2020

@author: andykim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:58:34 2020

@author: andykim
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy

#Test for 10 workers
w = 21

#number of signals for workers
n = 25
k = 2

#Parent-child pair
pc_dict = {0:[1,2], 1:[3,4], 2:[5,6]}

#Child-parent pair
cp_dict = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2}

#Set ground truth
gt = 3

#prior beliefs for both worker and platform
prior_set = [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
#prior_set = [1.0, 0.52, 0.48, 0.27, 0.25, 0.24, 0.24]

#platform epsilon
plat_epsilon = 0.95

#platform signals for random testing
plat_sigs_ran = [0]*len(prior_set)
plat_posts_ran = [0]*(2**3 - 1)
plat_posts_ran[0] = 1.0

#platform signals for epsilon-greedy testing
plat_sigs_eps = [0]*len(prior_set)
plat_posts_eps = [0]*(2**3 - 1)
plat_posts_eps[0] = 1.0

#platform signals for heuristic method testing
plat_sigs_heur = [0]*len(prior_set)
plat_posts_heur = copy.deepcopy(prior_set)

#ground truth posteriors with random testing
gtpost_ran = []

#ground truth posteriors with epsilon-greedy testing
gtpost_eps = []

#ground truth posteriors with heuristic-method testing
gtpost_heur = []

#worker lists
workers = []


w_type_list = []

wo_posts = [0.0]*(2**3-1)
wo_posts[0] = 1.0

def worker_type(typ):
    eps_set = []
    e_0 = 0
    e_1 = 0
    e_2 = 0
    if typ == "G":
        e_0 = np.divide(random.randrange(870,900),1000)
        e_1 = np.divide(random.randrange(940,960),1000)
        e_2 = np.divide(random.randrange(940,960),1000)
    elif typ == "C":
        e_1 = np.divide(random.randrange(850,870),1000)  
        e_0 = np.divide(random.randrange(910,930),1000)
        e_2 = np.divide(random.randrange(940,960),1000)
    else:
        e_1 = np.divide(random.randrange(940,960),1000)     
        e_2 = np.divide(random.randrange(850,870),1000) 
#        e_0 = np.divide(random.randrange(930,960),1000)
        e_0 = np.divide(random.randrange(910,930),1000)

    eps_set = [e_0,e_1,e_2]
    return eps_set

#Function that generates the alphas and betas for worker, it takes alphabeta set and epsilons
def worker_abs(arr,eps):
    for i in range(len(eps)):
        alpha = 1 - eps[i] + np.divide(eps[i],k)
        beta = np.divide(eps[i],k)
        arr.append((alpha,beta))
    return arr
        
#Function that generates the signals for workers, it takes alphabeta set, signal array, and index
def worker_sigs(arr,sigs,ind):
    sigs[2*ind+1] = arr[ind][0] * sigs[ind]
    sigs[2*ind+2] = sigs[ind] - sigs[2*ind+1]

    return sigs

#Function that generates the posterior beliefs for the worker, it takes alphabeta set, signal array, and posterior array
def worker_posts(arr,sigs):

    w_posts = []
    w_posts.append(1.0)
    
    for i in range(len(arr)):
        first = np.multiply(np.multiply(np.power(arr[i][0],sigs[2*i+1]),np.power(arr[i][1],sigs[2*i+2])),prior_set[2*i+1])
        second = np.multiply(np.multiply(np.power(arr[i][0],sigs[2*i+2]),np.power(arr[i][1],sigs[2*i+1])),prior_set[2*i+2])
        w_posts.append(np.multiply(w_posts[i],np.divide(first,first+second)))
        w_posts.append(np.multiply(w_posts[i],np.divide(second,first+second)))

    return w_posts

def approx_generate(approx,ab):
    for i in range(len(ab)):
        for j in range(len(ab[i])):
            if i >= 1:
                approx.append(ab[i][j]*ab[0][i-1])
            else:
                approx.append(ab[i][j])
    return approx

def plat_counts(posts,p_sigs,q):
    tmp_posts = [0.0]*len(posts)
    tmp_posts[0] = 1.0
    tmp_posts[2*q+1] = posts[2*q+1]
    tmp_posts[2*q+2] = posts[2*q+2]
    tmp_posts[q] = posts[2*q+1] + posts[2*q+2] 
    p_alpha = 1 - plat_epsilon + np.divide(plat_epsilon,k)
    p_beta = np.divide(plat_epsilon,k)
    
    if q == 0:
        diff2 = np.divide(math.log(tmp_posts[1])-math.log(tmp_posts[2])-math.log(prior_set[1])+math.log(prior_set[2]),math.log(p_alpha) - math.log(p_beta))
        diff2 = diff2
        theta1constant = 10
        p_1 = theta1constant
        p_2 = theta1constant - diff2
        p_3 = np.divide(p_1,2)
        p_4 = p_1 - p_3
        p_5 = np.divide(p_2,2)
        p_6 = p_2 - p_5
    else:
        leftover = 1.0 - posts[2*q+1] - posts[2*q+2]
        #To buffer the cases when leftover is less than 0
#        if leftover <= 0:
#            leftover = 0.000000000000001
        if q == 1:        
            tmp_posts[2] = leftover
            tmp_posts[5] = tmp_posts[2] * np.divide(prior_set[5], prior_set[2])
            tmp_posts[6] = tmp_posts[2] * np.divide(prior_set[6], prior_set[2])
#            for i in range(len(posts)):
#                if i not in [0,2,q,2*q+1,2*q+2]:
#                    tmp_posts[i] = np.multiply(leftover,prior_set[i])
        if q == 2:
            tmp_posts[1] = leftover
            tmp_posts[3] = tmp_posts[1] * np.divide(prior_set[3], prior_set[1])
            tmp_posts[4] = tmp_posts[1] * np.divide(prior_set[4], prior_set[1])
        
#        print("question")
#        print(q)
#        print("tmp_posts")
#        print(tmp_posts)
        diff2 = np.divide(math.log(tmp_posts[1])-math.log(tmp_posts[2])-math.log(prior_set[1])+math.log(prior_set[2]),math.log(p_alpha) - math.log(p_beta))
        diff4 = np.divide(math.log(tmp_posts[3])-math.log(tmp_posts[4])-math.log(prior_set[3])+math.log(prior_set[4]),math.log(p_alpha) - math.log(p_beta))
        diff6 = np.divide(math.log(tmp_posts[5])-math.log(tmp_posts[6])-math.log(prior_set[5])+math.log(prior_set[6]),math.log(p_alpha) - math.log(p_beta))
        theta1constant = 10
        diff2 = diff2
        diff4 = diff4
        diff6 = diff6
        
        p_1 = theta1constant
        p_2 = theta1constant - diff2
        p_3 = np.divide(theta1constant+diff4,2)
        p_4 = p_1 - p_3
        p_5 = np.divide(p_2+diff6,2)
        p_6 = p_2 - p_5
#        print(p_1)
#        print(p_2)
#        print(p_3)
#        print(p_4)
#        print(p_5)
#        print(p_6)
   
    p_sigs[1] += p_1
    p_sigs[2] += p_2
    p_sigs[3] += p_3
    p_sigs[4] += p_4
    p_sigs[5] += p_5
    p_sigs[6] += p_6
    return p_sigs

#Update the posteriors of platform using the accumulated platform signals
def update_platform(i,plat_sigs,plat_posts):
    p_alpha = 1 - plat_epsilon + np.divide(plat_epsilon,k)
    p_beta = np.divide(plat_epsilon,k)

    left_eq = np.multiply(np.multiply(np.power(p_alpha,plat_sigs[2*i+1]),np.power(p_beta,plat_sigs[2*i+2])),prior_set[2*i+1])
    right_eq = np.multiply(np.multiply(np.power(p_alpha,plat_sigs[2*i+2]),np.power(p_beta,plat_sigs[2*i+1])),prior_set[2*i+2])
    plat_posts[2*i+1] = np.multiply(np.divide(left_eq,left_eq+right_eq),plat_posts[i])
    plat_posts[2*i+2] = np.multiply(np.divide(right_eq,left_eq+right_eq),plat_posts[i])
    return plat_posts    

#Update the posteriors of platform using the acculated platform signals
def update_heuristic_platform(i,plat_sigs,plat_posts):
    p_alpha = 1 - plat_epsilon + np.divide(plat_epsilon,k)
    p_beta = np.divide(plat_epsilon,k)

    left_eq = np.multiply(np.multiply(np.power(p_alpha,plat_sigs[2*i+1]),np.power(p_beta,plat_sigs[2*i+2])),prior_set[2*i+1])
    right_eq = np.multiply(np.multiply(np.power(p_alpha,plat_sigs[2*i+2]),np.power(p_beta,plat_sigs[2*i+1])),prior_set[2*i+2])
    
    plat_posts[2*i+1] = np.multiply(np.divide(left_eq,left_eq+right_eq),plat_posts[i])
    plat_posts[2*i+2] = np.multiply(np.divide(right_eq,left_eq+right_eq),plat_posts[i])
    
    return plat_posts  

def heuristic_method(plat_signals,approx_wposts):
    post_set = []
    for q in range(3):
        psigs = copy.deepcopy(plat_signals)
        posts = copy.deepcopy(approx_wposts)
#        print("approximate worker posts")
#        print(posts)
#        print("psigs before")
#        print(psigs)
        psigs = plat_counts(approx_wposts,psigs,q)
#        print("psigs after")
#        print(psigs)
        for i in range(3):
            posts = update_heuristic_platform(i,psigs,posts)
#        print("Updated posteriors heuristic")
#        print(posts)
        post_set.append(posts)
    print("possible posteriors")
    print(post_set)
    return post_set

def heuristic_decision(post_set):
    entropy_set = []
    for i in range(len(post_set)):
        entropy = 0
        for j in range(3,7):
            entropy += np.multiply(post_set[i][j],np.log2(post_set[i][j]))
        entropy_set.append(-entropy)
    print("entropy set")
    print(entropy_set)
    return np.argmin(entropy_set)

    
#Create the worker list with evenly distributed
for i in range(w):
    if i >= 0 and i < 7:
        w_type_list.append("G")
    elif i >= 7 and i < 14:
        w_type_list.append("C")
    else:
        w_type_list.append("N")

random.shuffle(w_type_list)
#Main function for running 5 experiments asking question 1.

for i in range(w):
    w_type = w_type_list[i]
#    w_type = "N"
    w_eps = worker_type(w_type)
    abset = []
    abset = worker_abs(abset,w_eps)
    wo_approx = []
    wo_approx.append(1.0)
    wo_approx = approx_generate(wo_approx,abset)
    print(wo_approx)
    print("Current platform signal counts")
    print(plat_sigs_heur)
    post_set = heuristic_method(plat_sigs_heur,wo_approx)
    hq = heuristic_decision(post_set)
    print(hq)
    
    w_sigs = [0]*(2**len(w_eps)-1)
    w_sigs[0] = n
    
    for j in range(len(abset)):
        w_sigs = worker_sigs(abset,w_sigs,j)
    wo_posts = worker_posts(abset,w_sigs)
    
#    #Randomly choose question
    rq = random.choice([0,1,2])
##    #Question based on epsilon
    q = np.argmin(w_eps)

    plat_sigs_ran = plat_counts(wo_posts,plat_sigs_ran,rq)
    plat_sigs_eps = plat_counts(wo_posts,plat_sigs_eps,q)
    plat_sigs_heur = plat_counts(wo_posts,plat_sigs_heur,hq)

    for x in range(3):
        plat_posts_ran = update_platform(x,plat_sigs_ran,plat_posts_ran)
        plat_posts_eps = update_platform(x,plat_sigs_eps,plat_posts_eps)
        plat_posts_heur = update_platform(x,plat_sigs_heur,plat_posts_heur)

    workers.append(i+1)
#    if plat_posts_ran[gt] >= 0.99:
#        print("Random method converged at worker " + str(i))
#    if plat_posts_eps[gt] >= 0.99:
#        print("Greedy method converged at worker " + str(i))
#    if plat_posts_heur[gt] >= 0.99:
#        print("Heuristic method converged at worker " + str(i))
    
    gtpost_ran.append(plat_posts_ran[gt])
#    print("Greedy posteriors")
    gtpost_eps.append(plat_posts_eps[gt])
    print("final for random")
    print(plat_posts_ran[gt])
    print("Final probability for greedy")
    print(plat_posts_eps[gt])
    gtpost_heur.append(plat_posts_heur[gt])
    print("Final prob for heuristic posterior")
    print(plat_posts_heur[gt])

# Plot the platform posterior on ground truth
plt.plot(workers, gtpost_ran,label = 'Random')
plt.plot(workers, gtpost_eps, label = 'Greedy')
plt.plot(workers, gtpost_heur, label = 'Heuristic')
plt.xlabel('Number of Agents')
plt.ylabel('Platform Posterior on Ground Truth')
plt.legend()
# Add a legend
#plt.legend("Random","Epsilon-Greedy","Heuristic")
#plt.legend("Heuristic")

# Show the plot
plt.xticks(np.arange(min(workers), max(workers)+1, 1.0))
plt.show()


