#Biblioteka odpowiadajaca za rysowanie wykresu grafu
import matplotlib.pyplot as plt
#Biblioteka do badania wykresow i sieci
import networkx as nx
#Odpowiada za operacja na macierzach i grafach
import numpy as np
from itertools import chain
import pandas as pd
from networkx.algorithms import tournament
import time

'''
G = nx.Graph()

# Dodanie wierzcholkow grafu wraz z atrybutem demand
G.add_node('1')
G.add_node('2')
G.add_node('3')
G.add_node('4')
G.add_node('5')

G.add_edge('1', '2', weight = 10)
G.add_edge('1', '3', weight = 18)
G.add_edge('1', '4', weight = 20)
G.add_edge('1', '5', weight = 40)
G.add_edge('2', '3', weight = 35)
G.add_edge('2', '4', weight = 12)
G.add_edge('2', '5', weight = 15)
G.add_edge('3', '4', weight = 25)
G.add_edge('3', '5', weight = 10)
G.add_edge('4', '5', weight = 30)



adj = np.matrix([[0, 29, 82, 46, 68, 52, 72, 42, 51, 55],
                [29, 0, 55, 46, 42, 43, 43, 23, 23, 31],
                [82, 55, 0, 68, 46, 55, 23, 43, 41, 29],
                [46, 46, 68, 0, 82, 15, 72, 31, 62, 42],
                [68, 42, 46, 82, 0, 74, 23, 52, 21, 46],
                [52, 43, 55, 15, 74, 0, 61, 23, 55, 31],
                [72, 43, 23, 72, 23, 61, 0, 42, 23, 31],
                [42, 23, 43, 31, 52, 23, 42, 0, 33, 15],
                [51, 23, 41, 62, 21, 55, 23, 33, 0, 29],
                [55, 31, 29, 42, 46, 31, 31, 15, 29, 0]])
'''
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
'''
def generate_graph_matrix(n):
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    while True:
        distances = np.random.randint(1, 10, size=(n, n))  # Losowe generowanie wag krawędzi
        np.fill_diagonal(distances, 0)  # Wyzerowanie przekątnej (brak krawędzi między wierzchołkami)
        
        is_triangle_inequality_satisfied = True
        for i in range(n):
            for j in range(n):
                if i != j:
                    for k in range(n):
                        if k != i and k != j:
                            # Sprawdzanie nierówności trójkąta dla każdej trójki wierzchołków
                            if distances[i, j] + distances[j, k] < distances[i, k]:
                                is_triangle_inequality_satisfied = False
                                break
                    if not is_triangle_inequality_satisfied:
                        break
                if not is_triangle_inequality_satisfied:
                    break
            if not is_triangle_inequality_satisfied:
                break
        
        if is_triangle_inequality_satisfied:
            break
    
    return distances
'''

# Przykładowe użycie
'''

n = 25  # liczba wierzchołków
graph_matrix = generate_graph_matrix(n)
adj = graph_matrix
'''
tic()
n=30
adj = np.random.randint(low=1, high=50, size=(n, n))
np.fill_diagonal(adj, 0)

G = nx.from_numpy_matrix(adj)



print(G)
print(nx.to_numpy_matrix(G, weight='weight'))
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
print("Koszt trasy: ", suma)
toc()
'''
# Ustawienie pozycji wierzcholkow na wykresie
G.nodes['1']['pos'] = (2,2)
G.nodes['2']['pos'] = (0,0.5)
G.nodes['3']['pos'] = (1,-2)
G.nodes['4']['pos'] = (4,0.5)
G.nodes['5']['pos'] = (3,-2)

# Ustawienie podpisu wiercholkow na wykresie
G.nodes['1']['label_pos'] = (1.9,1.9)
G.nodes['2']['label_pos'] = (0.1,0.4)
G.nodes['3']['label_pos'] = (0.9,-1.9)
G.nodes['4']['label_pos'] = (3.9,0.4)
G.nodes['5']['label_pos'] = (2.9,-1.9)

node_pos = nx.get_node_attributes(G,'pos')
label_pos = nx.get_node_attributes(G,'label_pos')
wagi = nx.get_edge_attributes(G,'weight')
labels_G = dict()
for e in G.edges(): 
    labels_G[e] = '{}'.format(wagi[(e[0], e[1])])
labels_E = dict()
for e in E.edges(): 
    labels_E[e] = '{}'.format(wagi[(e[0], e[1])])


nx.draw_networkx(G, node_pos,node_color='white', node_size=800)
nx.draw_networkx_edge_labels(G, label_pos, edge_labels=labels_G, font_size=8)
plt.show()
nx.draw_networkx(E, node_pos,node_color='white', node_size=800)
nx.draw_networkx_edge_labels(G, label_pos, edge_labels=labels_E, font_size=8)
plt.show()


'''





