# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.svm import SVC #Classificador categorico, nao continuo. 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
da = load_iris()


#data = pd.DataFrame(da.data)  # carrega dos dados do .CSV, apenas parte de dados
#data.columns = da.feature_names # Carrega os nomes ddos atributos para colunas
#label = da.target # Carrega o target, valor preditivo

path = 'D:\Python\Classifier_CHAID/'
input_file = 'TML_test.csv'  #Verifique se a planilha está nos moldes corretos. 
dados = pd.read_csv(path+input_file)

data = dados
label = dados.f100 *10 # *TML: Artificio utilizado para sanar o problema do classificador SVM do qual os valores da predição DEVEM ser números inteiros

# PARÂMETROS DE CONFIGURAÇÃO DO ACO:

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)
#verificar a % destinada para TESTES e TREINO. 20% para teste

# PARÂMETROS DE CONFIGURAÇÃO DO ACO:

q0 = 0.7 # :coefiente [Exploration-Exploitation], um numero real no intervalo de [0, 1]. Par^ametro que de
#ne a regra de transic~ao de estado, entre uma busca gulosa ou probabilstica.
subset = 10 #Subconjunto de atributos selecionados
ants = 81 #quantidade de formigas, de acordo com a qtde de atributos
max_iter = 50 #parâmetro padrão para ACO
phe = np.random.uniform(0.1,1,81) #4 valores aleatorios entre 0,1 e 1. É um ARRAY (Lista)

"""
param ant_count:
param generations:
param alpha: relative importance of pheromone
param beta: relative importance of heuristic information
param rho: pheromone residual coefficient
param q: pheromone intensity
param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
"""        

for m in range(max_iter): #vai fazer 50 vezes 
# Calculando o caminho para cada formiga
    
    path =[] #Cria uma lista vazia, caminho 
    
    for j in np.arange(ants): #  Valores são gerados dentro de um intervalo, retorna um array 
                                   #ex: np.arange(4) array([0, 1, 2, 3])
        new_ph =np.copy(phe) # Pega uma cópia do array phe
        cityA=[] #declara uma lista vazia
        A = np.random.randint(0,81) # Pega um numero apenas inteiro entre 0 e 3
        cityA.append(A) # Adicionar o A na lista cityA, primeira posição
        for i in np.arange(subset-1): # Vai fazer uma lista arrange(2), = lista[0,1]
            r = np.random.uniform(0,1) # r recebe um numero aleatorio entre [0 - 1]    
            
            # Rotina de atualização do feromonio:
            if r < q0: # R tem 70% de chances de cair nesse IF // Sempre vai para a qtde de feromonio maior - Gulosa
                new_ph[A]=0 # vai atualizar a posição A na lista NEW_PH para 0 / Ele zera o caminho anterios para nao ter risco de retornar
                A = np.argmax(new_ph) # Retorna o ÌNDICE do maior valor de NEW_PH
                cityA.append(A) # Vai pegar o maior valor dete argumento e adicionar a cityA
            else: #30% de chances de entrar nesse ELSE - Busca probabilistica
                new_ph[A]=0 #Atualiza o valor do INDICE A para 0
                prob = new_ph / (sum(new_ph)) # Faz uma probabilidade - vai gerar um array de probabilidades media
                # np.random.choice(range(4),p=prob) # Faz uma esolha baseada em probabilidade
                A = np.random.choice(range(81),p=prob) # Faz uma esolha baseada em probabilidade
                cityA.append(A) #Vai adicionar o A  ex [0,1], [0,3] /um caminho
        path.append(cityA) #Vai adicionar o caminho dentro da lista -
                            #Passos que a formiga deu ex: [[0,1,3],[1,3,2],[.,.,.],[.,.,.]] retorna 4 caminhos 
         
    #Saiu do for j np.arrange
    #### Calcula a função objetivo, Fitness, para cada caminho atraves do SVM
    accuracy=[] # Cria uma lista vazia
    for k in np.arange(len(path)): #Vai pegar o tamanho do path, que tem 4 listas, logo = 4
        model = SVC(random_state=42) #setando o 42 randon state
        model.fit( X_train.iloc[:,path[k]],y_train) # Faz o FIT doa caminhos de cada um
        y_pred = model.predict(X_test.iloc[:,path[k]]) #y_Pred vai receber a predição do modelo SVM baseado no caminho
        acc= accuracy_score(y_test,y_pred) #Acuracia entre essas 2 variaveis - Pesquisar mais esta função
        accuracy.append(acc)  #Salva todas as acuracias na lista
    max_acc = np.max(accuracy)   #pega o valor máximo 
    max_acc_ind = np.argmax(accuracy) #pega o índice do VALOR MAX de acuracia
    best_atts = path[max_acc_ind] # Vai atribiuir o melhor caminho a partir do INDICE de maior acuracia 
    
    # Rotina de atualização do feromonio
    phe = phe* 0.8
    for l in np.arange(len(best_atts)): # se o caminho de melhor acuracia for [0,1,3], ele vai me retornar = [0,1,2]  
        q= (phe[best_atts[l]]*1.2) / 0.8 # atualiza o ferominio do melhor caminho encontrado 
        phe[best_atts[l]] =q # atualiza os valores dos feromonios nos INDICES de cada melhor caminho
    final_Attribtes = ((np.argsort(phe).tolist())[::-1])[:subset] #Ordena do maior para o menor dentro do feromonio

print("Os melhores atributos são:",final_Attribtes)
print("A melhor acurácia é:",max_acc*100)