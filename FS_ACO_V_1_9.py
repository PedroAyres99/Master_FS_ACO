# -*- coding: utf-8 -*-
"""
@author: Pedro Ayres

Descrição do código. 

"""

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.spatial.distance import cosine
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC #Classificador categorico, nao continuo.

import warnings
warnings.filterwarnings('ignore')

# CLASSE QUE REPRESENTA UM VÉRTICE (ATRIBUTO)

class Vertice:

    def __init__(self, numero):
        self.numero = numero
        self.feromonio = 0

    def obterNumero(self):
        return self.numero

    def obterFeronomio(self):
        return self.feromonio

    def setFeromonio(self, feromonio):
        self.feromonio += feromonio

# CLASSE QUE REPRESENTA UMA ARESTA EX: (1,3)

class Aresta:

    def __init__(self, origem, destino, custo):
        self.origem = origem
        self.destino = destino
        self.custo = custo
        self.feromonio = None

    def obterOrigem(self):
        return self.origem

    def obterDestino(self):
        return self.destino

    def obterCusto(self):
        return self.custo

    def obterFeronomio(self):
        return self.feromonio

    def setFeromonio(self, feromonio):
        self.feromonio = feromonio


# CLASSE QUE REPRESENTA UM GRAFO COMPLETO

class Grafo:

    def __init__(self, num_vertices):
        self.num_vertices = num_vertices  # número de vértices do grafo
        self.arestas = {}  # dicionário com as arestas
        self.vizinhos = {}  # dicionário com todos os vizinhos de cada vértice
        self.vertices = {}  # dicionário com os vértices(atributos)

    def adicionarAresta(self, origem, destino, custo):
        aresta = Aresta(origem=origem, destino=destino, custo=custo)
        self.arestas[(origem, destino)] = aresta
        if origem not in self.vizinhos:
            self.vizinhos[origem] = [destino]
        else:
            self.vizinhos[origem].append(destino)

    def adicionarVertice(self, numero): #Inserido
        vertice = Vertice(numero=numero)
        self.vertices[numero] = vertice

    def obterFeromonioVertice(self, numero): #Inserido
        return self.vertices[numero].obterFeronomio()

    def setFeromonioVertice(self, numero, feromonio): #Inserido
        self.vertices[numero].setFeromonio(feromonio)
        
    def obterCustoAresta(self, origem, destino):
        return self.arestas[(origem, destino)].obterCusto()

    def obterFeromonioAresta(self, origem, destino):
        return self.arestas[(origem, destino)].obterFeronomio()

    def setFeromonioAresta(self, origem, destino, feromonio):
        self.arestas[(origem, destino)].setFeromonio(feromonio)

    def obterCustoCaminho(self, caminho):
        custo = 0
        for i in range(self.num_vertices - 1):
            custo += self.obterCustoAresta(caminho[i], caminho[i + 1])

        # adiciona o último custo

        custo += self.obterCustoAresta(caminho[-1], caminho[0])
        return custo


class GrafoCompleto(Grafo):

# FUNÇÃO GERAR GRAFO COMPLETO

    def gerar(self, matrix):
        for i in range(1, self.num_vertices + 1):
            self.adicionarVertice(i) # Adiciona um vértice (Atributo)
            for j in range(1, self.num_vertices + 1):
                if i != j:
                    peso = matrix[i - 1][j - 1] # O peso será os valores da similaridade de cossenos
                    # peso = random.randint(1, 10) # maneira randômica
                    self.adicionarAresta(i, j, peso)


# CLASSE QUE REPRESENTA UMA FORMIGA

class Formiga:

    def __init__(self, cidade):
        self.cidade = cidade
        self.solucao = []
        self.custo = None

    def obterCidade(self):
        return self.cidade

    def setCidade(self, cidade):
        self.cidade = cidade

    def obterSolucao(self):
        return self.solucao

    def setSolucao(self, solucao, custo):

        # atualiza a solução

        if not self.custo:
            self.solucao = solucao[:]
            self.custo = custo
        else:
            if custo < self.custo:
                self.solucao = solucao[:]
                self.custo = custo

    def obterCustoSolucao(self):
        return self.custo


# CLASSE DO ACO

class ACO:

    def __init__(self, grafo, num_formigas, alfa=1.0, beta=5.0, iteracoes=100, evaporacao=0.2):
        self.grafo = grafo
        self.num_formigas = num_formigas
        self.alfa = alfa  # importância do feromônio
        self.beta = beta  # importância da informação heurística
        self.iteracoes = iteracoes  # quantidade de iterações
        self.evaporacao = evaporacao  # taxa de evaporação
        self.formigas = []  # lista de formigas

        lista_cidades = [cidade for cidade in range(1, self.grafo.num_vertices + 1)]

        # cria as formigas colocando cada uma em uma cidade

        for k in range(self.num_formigas):
            cidade_formiga = random.choice(lista_cidades)
            lista_cidades.remove(cidade_formiga)
            self.formigas.append(Formiga(cidade=cidade_formiga))
            if not lista_cidades:
                lista_cidades = [cidade for cidade in range(1, self.grafo.num_vertices + 1)]

        # calcula o custo guloso pra usar na inicialização de TODOS VALORES DE FEROMONIO

        custo_guloso = 0.0  # custo guloso
        vertice_inicial = random.randint(1, grafo.num_vertices)  # seleciona um vértice aleatório
        vertice_corrente = vertice_inicial
        visitados = [vertice_corrente]  # lista de visitados
        while True:
            vizinhos = (self.grafo.vizinhos[vertice_corrente])[:]
            (custos, escolhidos) = ([], {})
            for vizinho in vizinhos:
                if vizinho not in visitados:
                    custo = self.grafo.obterCustoAresta(vertice_corrente, vizinho)
                    escolhidos[custo] = vizinho
                    custos.append(custo)
            if len(visitados) == self.grafo.num_vertices:
                break
            #min_custo = min(custos)  # pega o menor custo da lista
            min_custo = max(custos)  # pega o maior custo da lista
            # Ao assumir a similaridade de cossenos, deveremos pegar o MAIOR INDICE -> VERIFICAR

            custo_guloso += min_custo  # adiciona o custo ao total
            vertice_corrente = escolhidos[min_custo]  # atualiza o vértice corrente
            visitados.append(vertice_corrente)  # marca o vértice corrente como visitado

        # adiciona o custo do último visitado ao custo_guloso

        custo_guloso += self.grafo.obterCustoAresta(visitados[-1], vertice_inicial)

# Dorigo propõe inicializar o feromônio de todas as arestas por 1 / (n.Ln),
# onde Lnn é o custo de uma construção puramente gulosa. CALCULADO ACIMA

        # inicializa o feromônio de todas as arestas
        for chave_aresta in self.grafo.arestas:
            #feromonio = 1.0 / (self.grafo.num_vertices * custo_guloso)
            feromonio = 0.2
            self.grafo.setFeromonioAresta(chave_aresta[0], chave_aresta[1], feromonio)
            
        # Sugestão de adotar o valor inicial de feromonio = 0,2 para todas arestas, seguindo o proposto em UFSACO
        # Feromonio inicializado com 0.0055 a partir da construção gulosa, para exemplo Bando de Dados: Wine
# NÚCLEO DO ACO:

    def rodar(self):

        for it in range(self.iteracoes):

            # lista de listas com as cidades visitadas por cada formiga
            cidades_visitadas = []
            for k in range(self.num_formigas):
                # adiciona a cidade de origem de cada formiga
                cidades = [self.formigas[k].obterCidade()]
                cidades_visitadas.append(cidades)

            # para cada formiga se constrói uma solução (CIDADES VISITADAS)
            for k in range(self.num_formigas):
                for i in range(1, self.grafo.num_vertices):
                    
                    # obtém todos os vizinhos que não foram visitados
                    cidades_nao_visitadas = list(set(self.grafo.vizinhos[self.formigas[k].obterCidade()]) - set(cidades_visitadas[k]))
                   
                    # somatório do conjunto de cidades não visitadas pela formiga "k"
                    # servirá para utilizar no cálculo da probabilidade. Primeiro calcula-se o SOMATORIO para depois a PROBABILIDADE 
                    somatorio = 0.0
                    for cidade in cidades_nao_visitadas:
                        # calcula o feromônio
                        feromonio = self.grafo.obterFeromonioAresta(self.formigas[k].obterCidade(), cidade)

                        # obtém a distância
                        distancia = self.grafo.obterCustoAresta(self.formigas[k].obterCidade(), cidade)

                        # adiciona no somatório 
                        #Parte de baixo da equação de probabilidade (1) descrita no artigo      
                        somatorio += (math.pow(feromonio, self.alfa) * math.pow(1.0 / distancia, self.beta))

                    # probabilidades de escolher um caminho
                    probabilidades = {}
                    for cidade in cidades_nao_visitadas:
                        # calcula o feromônio
                        feromonio = self.grafo.obterFeromonioAresta(self.formigas[k].obterCidade(), cidade)
						
                        # obtém a distância
                        distancia = self.grafo.obterCustoAresta(self.formigas[k].obterCidade(), cidade)
						
                        # obtém a probabilidade
                        probabilidade = (math.pow(feromonio, self.alfa) * math.pow(1.0 / distancia, self.beta)) / (somatorio if somatorio > 0 else 1)
						
                        # adiciona na lista de probabilidades
                        probabilidades[cidade] = probabilidade
                        
                    # obtém a cidade escolhida, de maior probabilidade
                    cidade_escolhida = max(probabilidades, key=probabilidades.get)
                    
                    # adiciona a cidade escolhida a lista de cidades visitadas pela formiga "k"
                    cidades_visitadas[k].append(cidade_escolhida)

                # atualiza a solução encontrada pela formiga
                self.formigas[k].setSolucao(cidades_visitadas[k], self.grafo.obterCustoCaminho(cidades_visitadas[k]))

            # Para cada formiga temos um caminho percorrido (CIDADES_VISITADAS)
            
            #Introduzir o modelador para avaliar cada SUBSET de soluções geradas por cada formiga
                
                
            #Introduzir um temporizador    
                
            # atualiza quantidade de feromônio

            for aresta in self.grafo.arestas:

                # somatório dos feromônios da aresta
                somatorio_feromonio = 0.0

                # para cada formiga "k" TODAS AS FORMIGAS NESTE CASO ATUALIZAM O FEROMONIO. NO RANKBASED apenas as TOP MELHORES são atualizadas
                # O RANK BASED apresenta um resultado melhor
                for k in range(self.num_formigas):
                    arestas_formiga = []

                    # gera todas as arestas percorridas da formiga "k"
                    for j in range(self.grafo.num_vertices - 1):
                        arestas_formiga.append((cidades_visitadas[k][j], cidades_visitadas[k][j + 1]))
                    # adiciona a última aresta
                    arestas_formiga.append((cidades_visitadas[k][-1], cidades_visitadas[k][0]))

                    # verifica se a aresta faz parte do caminho da formiga "k"
                    if aresta in arestas_formiga:
                        # O somatório será utilizado na equação de atualização do feromonio na aresta.
                        somatorio_feromonio += (1.0 / self.grafo.obterCustoCaminho(cidades_visitadas[k]))

                # seta o VÉRTICE, unidade contadora para cada VERTICE(ATRIBUTO) presente no par de arestas        
                #if aresta in arestas_formiga:
                #    self.grafo.setFeromonioVertice(aresta[0], 1)
                #    self.grafo.setFeromonioVertice(aresta[1], 1)
                    
                #ATUALIZAÇÃO DO FEROMONIO NA ARESTA, LEVA EM CONSIDERAÇÃO A EQUAÇÃO PROPOSTA NO ANT SYSTEM
                # calcula o novo feromônio, PARA A ARESTA APOS ANALISE DE PERCURSO DE FORMIGAS
                novo_feromonio = (1.0 - self.evaporacao) * self.grafo.obterFeromonioAresta(aresta[0], aresta[1]) + somatorio_feromonio
				
                # seta o novo feromônio da aresta
                self.grafo.setFeromonioAresta(aresta[0], aresta[1], novo_feromonio)


        # FIM DO CICLO DAS ITERAÇÕES
        # percorre para obter as soluções das formigas
        (solucao, custo) = (None, None)
        for k in range(self.num_formigas):
            if not solucao:
                solucao = self.formigas[k].obterSolucao()[:]
                custo = self.formigas[k].obterCustoSolucao()
            else:
                aux_custo = self.formigas[k].obterCustoSolucao()
                if aux_custo < custo:
                    solucao = self.formigas[k].obterSolucao()[:]
                    custo = aux_custo
        print('Solução final: %s | custo: %d\n' % (' -> '.join(str(i) for i in solucao), custo))
        
        #Salvar os dados em uma planilha para depois dinamicamente realizar os testes
        #
#------------------------------------------------------------------------------------------------------#
# INICIO DO CÓDIGO
# IMPORTANDO O BANCO DE DADOS

da = load_wine()

data = pd.DataFrame(da.data)
data_2 = da.data
data.columns = da.feature_names
label = da.target

print (data.head())
print (data.shape)


# MATRIZ SIMILARIDADE DE COSSENOS:

# FUNÇÃO QUE CALCULA SIMILARIDADE ENTRE 2 PARES DE ATRIBUTOS:
# A distância entre os nós do GRAFO TSP é a distância. Em nosso caso, assumiremos a SIMILARIDADE DE COSSENOS
# (MATRIZ DE DISTANCIA)

# Função que calcula a distancia de cossenos

def cosine_distance(v1, v2):
    (v1_abs, v2_abs) = ([], [])
    for i in range(0, len(v1)):
        v1_abs.append(v1[i] * 100.0 / (v1[i] + v2[i] or 1))
        v2_abs.append(v2[i] * 100.0 / (v1[i] + v2[i] or 1))

    return 1 / (1 - spatial.distance.cosine(v1_abs, v2_abs))  # Inserido o artificio: (1 / (-1...)), cosine_similarity = 1 - cosine_distance.
    #return 1 - spatial.distance.cosine(v1_abs, v2_abs)  # Inserido o artificio: (1 / (-1...)), cosine_similarity = 1 - cosine_distance.

# Criando uma lista vazia

matrix = np.zeros((data.shape[1], data.shape[1]))

# matrix = (cosine_similarity(data, data))

# INTERAGINDO AO LONGO Da matriz de distância PARA calcular AS DISTANCIAS entre pares de NóS

for k in range(len(data.columns)):
    data_1 = data.iloc[:, [k]].values
    for j in range(len(data.columns)):
        data_2 = data.iloc[:, [j]].values
        matrix[k, j] = cosine_distance(data_1, data_2)
        j += 1
    k += 1

df_matrix_similaridade = pd.DataFrame(matrix, columns=data.columns,
        index=data.columns)

# SALVANDO A MATRIZ:
# df_matrix_similaridade.to_csv('Matriz_Similaridade.csv')
# ------------------------------------------------------------------------------------------------

num_vertices = 13

# num_formigas = 13 #Dessassociar pois o numero de formigas pode ser diferente do Nº de VERTICES
# print('Teste de grafo com %d vertices...\n' % num_vertices)
# Nº de Iterações pode ser uma variavel informada.

# GERAR O GRAFO COMPLETO (m x n), EX WINE: 156:

grafo_completo = GrafoCompleto(num_vertices=num_vertices)
grafo_completo.gerar(matrix)

# CLASSE ACO A PARTIR DO GRAFO E PARÂMETROS

"""
OBSERVAÇÕES RELACIONADAS AOS PARÂMETROS DO ACO: 
- Dorigo propoe que alpha deve ficar em torno de 1. Um valor muito baixo de 
provocaria estagnação precoce do algoritmo e um valor alto demais o aproxima de uma construção gulosa. 
O ajuste deste parâmetro é importante para se obter uma boa diversificação, propõe que fique em torno de 5.

 - UFSACO: (NCmax=50), theinitialamountofpheromoneforeachfeatureis 0.2(τi=0.2), 
#pheromoneevaporationcoefficient issetto 0.2, parameter β is setto1(β=1)

"""

aco2 = ACO(grafo=grafo_completo, num_formigas=grafo_completo.num_vertices, alfa=1, beta=5, iteracoes=100,
    evaporacao=0.2)
aco2.rodar()
