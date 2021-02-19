# -*- coding: utf-8 -*-
"""
@author: Pedro Ayres, Jodelson Sabino, Bruno Nazário

O código abaixo consiste na Seleção de Atributos de determinado Banco de Dados baseado na similaridade de cossenos 
entre 2 atributos. É utilizado a otimização por colônia de formiga para avalizar subsets através de uma modelagem 

Feature Selection (Seleção de Atributos): consiste em algoritmos com propósitos de reduzir a dimensionalidade 
de banco de dados, selecionando sub-conjuntos relevantes para a construção de modelos preditivos. 
Atributos redundantes prejudicam a performance do algoritmo de aprendizagem de máquina, 
logo selecionar de maneira apropriada os atributos de entrada auxilia no desempenho da classificação.

■ Keywords: Feature Selection; Ant Colony Optimization; Dimensionality Reduction; Data Classification; Wrapper Methods; Python.
"""

import numpy as np
import pandas as pd
import random
import math
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from scipy import spatial
from scipy.spatial.distance import cosine
from sklearn import svm #Classificador categorico, nao continuo.

import warnings
warnings.filterwarnings('ignore')

# CLASSE QUE REPRESENTA O MELHOR CUSTO DE CADA ITERACAO
        
class Acuracia:

    def __init__(self, iteracao):
        self.solucao = []
        self.acuracia = None

    def setSolucao(self, solucao, acuracia):
        self.solucao = solucao
        self.acuracia = acuracia
    
    def obterAcuracia_final(self):
        return self.acuracia

    def obterSolucao_final(self):
        return self.solucao

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
        return self.custo # O custo da ARESTA SERÁ O PESO (SIMILARIDADE DE COSSENOS)

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
        for i in range(self.num_vertices):
            #self.adicionarVertice(i) # Adiciona um vértice (Atributo)
            for j in range(self.num_vertices):
                if i != j:
                    peso = matrix[i][j] # O peso será os valores da similaridade de cossenos
                    # peso = random.randint(1, 10) # maneira randômica
                    self.adicionarAresta(i, j, peso)

# CLASSE QUE REPRESENTA UMA FORMIGA

class Formiga:

    def __init__(self, cidade):
        self.cidade = cidade
        self.solucao = []
        self.custo = None
        self.acuracia = None

    def obterCidade(self):
        return self.cidade

    def setCidade(self, cidade):
        self.cidade = cidade

    def obterSolucao(self):
        return self.solucao

    def setSolucao(self, solucao, acuracia):

        # atualiza a solução

        if not self.acuracia:
            self.solucao = solucao[:]
            self.acuracia = acuracia
        else:
            if acuracia < self.acuracia:
                self.solucao = solucao[:]
                self.acuracia = acuracia
    
    def obterCustoSolucao(self):
        return self.custo
    
    def obterAcuracia(self):
        return self.acuracia

# CLASSE DA ACO

class ACO:

    def __init__(self, grafo, num_formigas, alfa=1.0, beta=5.0, iteracoes=10, evaporacao=0.2, num_FS=8):
        self.grafo = grafo
        self.num_formigas = num_formigas
        self.alfa = alfa  # importância do feromônio
        self.beta = beta  # importância da informação heurística
        self.iteracoes = iteracoes  # quantidade de iterações
        self.evaporacao = evaporacao  # taxa de evaporação
        self.num_FS = num_FS  # Número de atributos a serem selecionados (Feature Selected)
        self.formigas = []  # lista de formigas
        self.acuracias = []  # lista de acuracias

        # cria as formigas alocando aleatoriamente cada em uma cidade(vértice)
        lista_cidades = [cidade for cidade in range(self.grafo.num_vertices)]
        for k in range(self.num_formigas):
            cidade_formiga = random.choice(lista_cidades)
            lista_cidades.remove(cidade_formiga)
            self.formigas.append(Formiga(cidade=cidade_formiga))
            if not lista_cidades:
                lista_cidades = [cidade for cidade in range(self.grafo.num_vertices)]

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
            min_custo = min(custos)  # pega o menor custo da lista


            custo_guloso += min_custo  # adiciona o custo ao total
            vertice_corrente = escolhidos[min_custo]  # atualiza o vértice corrente
            visitados.append(vertice_corrente)  # marca o vértice corrente como visitado
        # adiciona o custo do último visitado ao custo_guloso
        custo_guloso += self.grafo.obterCustoAresta(visitados[-1], vertice_inicial)

# Dorigo propõe inicializar o feromônio de todas as arestas por 1 / (n.Ln),
# onde Lnn é o custo de uma construção puramente gulosa. CALCULADO ACIMA

        # INICIALIZAÇÃO DO FEROMONIO EM TODAS AS ARESTAS
        for chave_aresta in self.grafo.arestas:
            feromonio = 1.0 / (self.grafo.num_vertices * custo_guloso)
            #feromonio = 0.2
            self.grafo.setFeromonioAresta(chave_aresta[0], chave_aresta[1], feromonio)
            
        # Sugestão de adotar o valor inicial de feromonio = 0,2 para todas arestas, seguindo o proposto em UFSACO
        # P/ ex: BD_Wine, Feromonio inicializado com 0.0055 a partir da construção gulosa.
    
    def imprimir(self):
        
        string = "\nSeleção de Atributos baseado na Otimização por Colônia de Formigas:"
        string += "\nProjetado para selecionar atributos de determinado BD através da ACO adotando a similaridade de cosseno entre pares de atributos como peso. A performance dos subsets(acuracia) através de uma modelagem será avaliada e ao final apresentado o conjunto de maior valor. A atualização do feromônio e regra de probabilidade foram desenvolvidas conforme algoritmo Ant-System"
        string += "\n--------------------"
        string += "\nParâmetros ACO:"
        string += "\nNúmero de Formigas:\t\t\t\t\t{}".format(self.num_formigas)
        string += "\nTaxa de evaporação:\t\t\t\t\t{}".format(self.evaporacao)
        string += "\nAlpha Heuristic(importância do feromônio):\t\t{}".format(self.alfa)
        string += "\nBeta Heuristic(importância da informação heurística):\t{}".format(self.beta)
        string += "\nNº de Iterações:\t\t\t\t\t{}".format(self.iteracoes)
        string += "\nNº de Atributos a serem selecionados:\t\t\t{}".format(self.num_FS)
        string += "\n--------------------"
        #string += "\nObservações:"
        #string += "\nO número de formigas influencia em quantos caminhos serão explorados a cada iteração."
        #string += "\nHeurísticas Alpha e Beta afetam a influência dos feromônios ou da distância heurística nas decisões das formigas."
        #string += "\nBeta reduz a influência da heurística ao longo do tempo. "
        #string += "\n--------------------"
        
        print(string)
        
# NÚCLEO DO ACO:

    def rodar(self, banco_dados, target):

       #INICIO DAS ITERACOES (NÚCLEO ACO):
        
        for it in range(self.iteracoes):
            
            # Cria lista com cidades a serem visitadas por cada formiga
            cidades_visitadas = []
            
            # adiciona a cidade de origem de cada formiga a lista
            for k in range(self.num_formigas):
                cidades = [self.formigas[k].obterCidade()]
                cidades_visitadas.append(cidades)

            # Para cada formiga se constrói uma solução, Listagem de CIDADES VISITADAS
            for k in range(self.num_formigas):
                
                # Etapa onde calcula-se a probabilidade dos vértices(atributos) vizinhos e adiciona 
                # a cidade escolhida à lista de cidades visitadas pela formiga "k"
                for i in range(1, self.grafo.num_vertices):
                    
                    # obtém todos os vizinhos que não foram visitados
                    cidades_nao_visitadas = list(set(self.grafo.vizinhos[self.formigas[k].obterCidade()]) - set(cidades_visitadas[k]))
                    
                    # somatório do conjunto de cidades não visitadas pela formiga "k"
                    # servirá para utilizar no cálculo da probabilidade. Primeiro calcula-se o SOMATORIO para depois a PROBABILIDADE 
                    somatorio = 0.0
                    for cidade in cidades_nao_visitadas:
                        # calcula o feromônio
                        feromonio = self.grafo.obterFeromonioAresta(self.formigas[k].obterCidade(), cidade)
                        # obtém a distância(Similiridade de Cossenos entre um vértice ex: (1,8)
                        distancia = self.grafo.obterCustoAresta(self.formigas[k].obterCidade(), cidade)
                        # adiciona no somatório 
                        #Denominador da equação de probabilidade (1) descrita no artigo      
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
                    
            # Fim de UMA ITERAÇÃO entre TODAS FORMIGAS, obtendo-se uma lista de caminhos percorridos por cada formiga        
            
            cidades_visitadas_PD = pd.DataFrame(cidades_visitadas)
            
            #CRIA-SE LISTAGEM DE CAMINHOS A PARTIR DA QTDE DE ATRIBUTOS QUE DESEJA-SE SELECIONAR: NUM_FS
            Lista_FS = cidades_visitadas_PD.iloc[:, 0:self.num_FS].values
            
            #Introduzir modelador para avaliar cada SUBSET de soluções geradas a partir de cada formiga
            # A função objetivo(Custo, no caso métrica ACURACIA) para cada caminho(SUBSETS) atraves de classificadores
            
            # SEPARANDO PARA TREINAMENTO / TESTE
            X_train, X_test, y_train, y_test = train_test_split(banco_dados, target, test_size=0.20, random_state=42)
                        
            #CLASSIFICAÇÃO REALIZADA ATRAVES DO SVM, ADOTANDO-SE AS COLUNAS SELECIONADAS NO SUBSET(NUM_FS)
            #accuracy=[] # Cria uma lista vazia
            for x in range(self.num_formigas): # Quantidade de formigas
                #Create a svm Classifier
                clf = svm.SVC(kernel='linear') # Linear Kernel
                clf.fit(X_train.iloc[:,Lista_FS[x]],y_train) # Aprender a base de treino, usar variaveis TREINOS
                #Predict the response for test dataset
                y_pred = clf.predict(X_test.iloc[:,Lista_FS[x]]) # Prevê a base de teste
                # Model Accuracy: how often is the classifier correct?
                #accuracy.append(accuracy_score(y_test, y_pred)) // Descomente para caso queira ver a listagem de acc
                # atualiza a SOLUÇÃO E ACURÁCIA encontrada pela formiga
                self.formigas[x].setSolucao(Lista_FS[x], accuracy_score(y_test, y_pred))


            # VERIFICANDO DENTRE TODAS AS FORMIGAS QUAL OBTEVE A MELHOR ACURACIA NA ITERACAO
            top_solucao = []
            top_acc = None
            for k in range(self.num_formigas):
                if not top_acc:
                    top_acc = self.formigas[k].obterAcuracia()
                else:   
                    aux_acc = self.formigas[k].obterAcuracia()
                    if aux_acc > top_acc:
                        top_solucao = self.formigas[k].obterSolucao()
                        top_acc = aux_acc
                        
            # SALVANDO A MAIOR ACURACIA E SEU RESPECTIVO CAMINHO DA ITERACAO NA LISTA DE ACURACIAS
            self.acuracias.append(Acuracia(iteracao = it))
            self.acuracias[it].setSolucao(solucao = top_solucao, acuracia = top_acc)

            # ROTINA DE ATUALIZAÇÃO DO FEROMÔNIO
            for aresta in self.grafo.arestas:

                # somatório dos feromônios da aresta
                somatorio_feromonio = 0.0

                # para cada formiga "k" TODAS AS FORMIGAS NESTE CASO ATUALIZAM O FEROMONIO.
                # O RANK BASED apresenta um resultado melhor / apenas as TOP MELHORES são atualizadas
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

                #ATUALIZAÇÃO DO FEROMONIO NA ARESTA, LEVA EM CONSIDERAÇÃO A EQUAÇÃO PROPOSTA NO ANT SYSTEM PAG:28
                # calcula o novo feromônio, PARA A ARESTA APOS ANALISE DE PERCURSO DE FORMIGAS
                novo_feromonio = (1.0 - self.evaporacao) * self.grafo.obterFeromonioAresta(aresta[0], aresta[1]) + somatorio_feromonio
				# O Algoritmo está atualizando toda a lista de cidades visitadas, e não apenas as selecionadas 
                # para a etapa de classificação, visando completar todo o depósito/evaporação no restante dos atributos 
                # seta o novo feromônio da aresta
                self.grafo.setFeromonioAresta(aresta[0], aresta[1], novo_feromonio)
            # FIM DA ROTINA DE ATUALIZAÇÃO DO FEROMÔNIO
                
        # FIM DO CICLO DAS ITERAÇÕES

        # PERCORRE LISTAGEM DE ACURACIAS QUAL DENTRE OBTEVE MAIOR VALOR, APRESENTANDO O SEU SUBSET 
        solucao_final = []
        acc_final = None
        #(solucao, custo) = (None, None)
        for k in range(self.iteracoes):
            if not acc_final:
                solucao_final = self.acuracias[k].obterSolucao_final()[:]
                acc_final = self.acuracias[k].obterAcuracia_final()
            else:
                aux_acc = self.acuracias[k].obterAcuracia_final()
                if aux_acc > acc_final:
                    solucao_final = self.acuracias[k].obterSolucao_final()[:]
                    acc_final = self.acuracias[k].obterAcuracia_final()
            
        
        print('Solução(sub-set) de atributos que apresentaram maior acurácia ao longo de', self.iteracoes, 'iterações:')
        print('%s | Acuracia: %d\n' % (' -> '.join(str(i) for i in solucao_final), acc_final))
        
        print("\n--------------------")        
        end_time = time.monotonic()
        print('Tempo de execução do código: ', timedelta(seconds=end_time - start_time))

#------------------------------------------------------------------------------------------------------#
#############################
##### INÍCIO DO CÓDIGO ######
#############################
        
# IMPORTANDO O BANCO DE DADOS

da = load_wine()

data = pd.DataFrame(da.data)
data.columns = da.feature_names
label = da.target

print('Informações do Bando de Dados(Amostras, Atributos):', data.shape)
#print (data.head())
#print (data.shape)

start_time = time.monotonic()

# ---------------------------------------------------------------------------------------#
# MATRIZ SIMILARIDADE DE COSSENOS:

# Função que calcula a similaridade entre par de atributos:
# O peso entre nós do GRAFO TSP(Caixeiro) é a distância. Em nosso caso, assumiremos a SIMILARIDADE DE COSSENOS
def cosine_distance(v1, v2):
    (v1_abs, v2_abs) = ([], [])
    for i in range(0, len(v1)):
        v1_abs.append(v1[i] * 100.0 / (v1[i] + v2[i] or 1))
        v2_abs.append(v2[i] * 100.0 / (v1[i] + v2[i] or 1))

    return 1 / (1 - spatial.distance.cosine(v1_abs, v2_abs))  # Inserido o artificio: (1 / (-1...)), cosine_similarity = 1 - cosine_distance.

# Criando uma matriz vazia
matrix = np.zeros((data.shape[1], data.shape[1]))

# Interagindo ao longo da matriz para calcular a similaridade de cossenos entre atributos (Vértices)
for k in range(len(data.columns)):
    data_1 = data.iloc[:, [k]].values
    for j in range(len(data.columns)):
        data_2 = data.iloc[:, [j]].values
        matrix[k, j] = cosine_distance(data_1, data_2)
        j += 1
    k += 1

df_matrix_similaridade = pd.DataFrame(matrix, columns=data.columns, index=data.columns)
# Salvando a matriz:
# df_matrix_similaridade.to_csv('Matriz_Similaridade.csv')

# ---------------------------------------------------------------------------------------#
# GERAÇÃO DO GRAFO COMPLETO
num_vertices = 13
# O código foi desenvolvido para assumir o NUM_FORMIGAS = NUM_VERTICES, em futura revisão, desassociar estas variáveis.

# ex: P/ BD Wine, GRAFO COMPLETO (13 x 13): 169 - 13(idênticas) = 156 Arestas
grafo_completo = GrafoCompleto(num_vertices=num_vertices)
grafo_completo.gerar(matrix)

# ---------------------------------------------------------------------------------------#
# CLASSE ACO A PARTIR DO GRAFO E PARÂMETROS

"""
OBSERVAÇÕES RELACIONADAS AOS PARÂMETROS DO ACO: 
- Dorigo propoe que alpha deve ficar em torno de 1. Um valor muito baixo de provocaria estagnação precoce 
do algoritmo e um valor alto demais o aproxima de uma construção gulosa. 
O ajuste deste parâmetro é importante para se obter uma boa diversificação, propõe que fique em torno de 5.
 - UFSACO: (NCmax=50), theinitialamountofpheromoneforeachfeatureis 0.2(τi=0.2), 
#pheromoneevaporationcoefficient issetto 0.2, parameter β is setto1(β=1)
 - WFACOFS: n Number of ants 10 / a Exploitation balance factor 1 / b Exploration balance factor 1
Iteration Number of iterations 20 / d Pheromone evaporation factor 0.15

"""
aco2 = ACO(grafo=grafo_completo, num_formigas=grafo_completo.num_vertices, alfa=1, beta=5, iteracoes=10,
    evaporacao=0.2, num_FS=8)

aco2.imprimir()

aco2.rodar(banco_dados = data, target = label)
