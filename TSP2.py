#Biblioteka odpowiadajaca za rysowanie wykresu grafu
import matplotlib.pyplot as plt
#Biblioteka do badania wykresow i sieci
import networkx as nx
#Odpowiada za operacja na macierzach i grafach
import numpy as np
from itertools import chain
import pandas as pd
from networkx.algorithms import tournament
import copy
import time

def TicTocGenerator():
    ti = 0          
    tf = time.time() 
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():

    toc(False)
'''
G = nx.Graph()

# Dodanie wierzcholkow grafu wraz z atrybutem demand
G.add_node('1')
G.add_node('2')
G.add_node('4')
G.add_node('3')
G.add_node('5')

G.add_edge('1', '2', weight = 10)
G.add_edge('1', '4', weight = 18)
G.add_edge('3', '1', weight = 20)
G.add_edge('2', '4', weight = 35)
G.add_edge('2', '3', weight = 12)
G.add_edge('4', '3', weight = 25)
G.add_edge('4', '5', weight = 10)
G.add_edge('3', '5', weight = 30)
G.add_edge('2', '5', weight = 15)
G.add_edge('1', '5', weight = 40)



adj=np.matrix([[    0,   153,   510,   706,   966,   581,   455,    70,   160,   372,   157,   567,   342,   398],
 [  153,     0,   422,   664,   997,   598,   507,   197,   311,   479,   310,   581,   417,   376],
 [  510,   422,     0,   289,   744,   390,   437,   491,   645,   880,   618,   374,   455,   211],
 [  706,   664,   289,     0,   491,   265,   410,   664,   804,  1070,   768,   259,   499,   310],
 [  966,   997,   744,   491,     0,   400,   514,   902,   990,  1261,   947,   418,   635,   636],
 [  581,   598,   390,   265,   400,     0,   168,   522,   634,   910,   593,    19,   284,   239],
 [  455,   507,   437,   410,   514,   168,     0,   389,   482,   757,   439,   163,   124,   232],
 [   70,   197,   491,   664,   902,   522,   389,     0,   154,   406,   133,   508,   273,   355],
 [  160,   311,   645,   804,   990,   634,   482,   154,     0,   276,    43,   623,   358,   498],
 [  372,   479,   880,  1070,  1261,   910,   757,   406,   276,     0,   318,   898,   633,   761],
 [  157,   310,   618,   768,   947,   593,   439,   133,    43,   318,     0,   582,   315,   464],
 [  567,   581,   374,   259,   418,    19,   163,   508,   623,   898,   582,     0,   275,   221],
 [  342,   417,   455,   499,   635,   284,   124,   273,   358,   633,   315,   275,     0,   247],
 [  398,   376,   211,   310,   636,   239,   232,   355,   498,   761,   464,   221,   247,     0]])

'''
'''
adj=np.matrix([[0, 29, 82, 46, 68, 52, 72, 42, 51, 55],
                [29, 0, 55, 46, 42, 43, 43, 23, 23, 31],
                [82, 55, 0, 68, 46, 55, 23, 43, 41, 29],
                [46, 46, 68, 0, 82, 15, 72, 31, 62, 42],
                [68, 42, 46, 82, 0, 74, 23, 52, 21, 46],
                [52, 43, 55, 15, 74, 0, 61, 23, 55, 31],
                [72, 43, 23, 72, 23, 61, 0, 42, 23, 31],
                [42, 23, 43, 31, 52, 23, 42, 0, 33, 15],
                [51, 23, 41, 62, 21, 55, 23, 33, 0, 29],
                [55, 31, 29, 42, 46, 31, 31, 15, 29, 0]])


adj=np.matrix([[0, 42, 6, 28, 35], 
                [9, 0, 29, 35, 38],
                [22, 13, 0, 29, 34],
                [17, 21, 14, 0, 2],
                [23, 24, 31, 5, 0]])





adj = np.matrix([[ 0, 10, 18, 20, 40],
                [10,  0, 35, 12, 15],
                [18, 35,  0, 25, 10],
                [20, 12, 25,  0, 30],
                [40, 15, 10, 30, 0]])



adj = np.matrix([[ 0, 20, 30, 10, 11],
                [15,  0, 16, 4, 2],
                [3, 5,  0, 2, 4],
                [19, 6, 18,  0, 30],
                [16, 4, 7, 16, 0]])

adj = np.matrix([[0, 10, 15, 20], 
                 [10, 0, 35, 25],
                 [15, 35, 0, 30], 
                 [20, 25, 30, 0]])
'''

n=7
adj = np.random.randint(low=1, high=500, size=(n, n))
np.fill_diagonal(adj, 0)

print(adj)

tic()
G = nx.from_numpy_matrix(adj)



print(G)
print(nx.to_numpy_matrix(G, weight='weight'))
adj = nx.to_numpy_matrix(G, weight='weight')
print(adj)
drzewo = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
E = nx.Graph(drzewo)
#print(G.edges())
#print(E.edges())
#print(E.nodes())

lista2 = list(nx.dfs_edges(E))
print(lista2)
print(list(chain(*lista2)))
listaW = pd.unique(list(chain(*lista2)))
print( "Optymalna ścieżka: ", listaW)

suma = 0
for i in range(0,len(listaW)-1):
    suma += G[listaW[i]][listaW[i+1]]['weight']

suma += G[listaW[0]][listaW[len(listaW)-1]]['weight']
print("Koszt trasy MST-DFS: ", suma)
toc()

tic()
adj = adj.astype(float)
N = adj.shape[1]
#print(N)
#print(adj)
infi = float('inf')


for i in range(N):
    adj[i,i] += float('inf')
#print(adj)

def redukcja_kosztow2(adj):
    R=0
    #Odejmowanie mimimum w wierszach
    for i in range(N):
        minimum = adj[i].min()
        if minimum != float('inf'):
            R += adj[i].min()
            for j in range(N):
                adj[i,j]-= minimum
    #Odejmowanie mimimum w kolumnach
    adj = adj.transpose()     
    for i in range(N):
        minimum = adj[i].min()
        if minimum != float('inf'):
            R += adj[i].min()
            for j in range(N):
                adj[i,j]-= minimum
    adj = adj.transpose() 
    return adj, R
    
adj, R = redukcja_kosztow2(adj)
#print(adj, R)
UB = float('inf')

def infdrzewo(adj, nmin, listanmin, LB, UB, n):
    adjnew = copy.deepcopy(adj) 

    lista = []
    listawierzcholkow = []
    listanmin.append(nmin)
    for i in range(1,N):
        if i not in listanmin and i != n:
            tempadj = copy.deepcopy(adj) 
            for j in range(0,N):
                tempadj[nmin,j] += float('inf')
                tempadj[j,i] += float('inf')
                tempadj[i,0] += float('inf')
            lista.append(tempadj)
            listawierzcholkow.append(i)

  
    listaR=[]
    listaa=[]
    cost = []
    templista = copy.deepcopy(lista) 
    for i in range(len(lista)):
        t1,t2 = redukcja_kosztow2(templista[i]) 
        listaa.append(t1)
        listaR.append(t2)
        cost.append(LB+t2+adjnew[nmin,listawierzcholkow[i]])

    nmin = listawierzcholkow[cost.index(min(cost))]
    UB = cost[-1]
    
    LB = min(cost)

    nadj = listaa[cost.index(min(cost))]
    adjnew = nadj
    return adjnew, cost, lista, listaa, listaR, nmin, listanmin, LB, UB

wyniki1 = []
wyniki2 = []

for i in range(0,N):
    adjnew, cost, lista, listaa, listaR, nmin, listanmin, LB, UB  = infdrzewo(adj,0, [], R, UB, i)
    costog = cost
    #print(cost)


    while(len(listanmin)<N-1 ):
        adjnew, cost, lista, listaa, listaR, nmin, listanmin, LB, UB = infdrzewo(adjnew, nmin, listanmin, LB, UB, 0)
        #print(adjnew)
        #print("mincost", min(cost))
        #print("UB", UB)
        #print(cost)
        #if min(costog)>UB:
            #break

 
    tlista = []
    for i in range(0,N):
        tlista.append(i)

    wl = (set(tlista) - set(listanmin)).pop()
    listanmin.append(wl)
    listanmin.append(listanmin[0])
    #print("Optymalna ścieżka: ", listanmin)
    #print("Minimalny koszt", cost)
    wyniki1.append(listanmin)
    wyniki2.append(cost)
    costog.remove(min(costog))
    #print("costog", costog)
    if min(costog)>UB:
        break


print("Optymalna ścieżka B%B: ", wyniki1[wyniki2.index(min(wyniki2))])
print("Koszt trasy: ", min(wyniki2))

toc()

    