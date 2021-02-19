# SELECÃO DE ATRIBUTOS BASEADO NO ALGORITMO DE OTIMIZACÃO COLÔNIA DE FORMIGAS PARA PROCESSOS MINERADORES

Repositório referente à pesquisa entitulada acima. Programa: Mestrado Profissional em Instrumentação, Controle e Automação de Processos de Mineração - PROFICAM. Instituições: UFOP / Instituto Tecnológico VALE - ITV.

Feature Selection (Seleção de Atributos): consiste em algoritmos com propósitos de reduzir a dimensionalidade de banco de dados, 
selecionando sub-conjuntos relevantes para a construção de modelos preditivos. Atributos redundantes prejudicam a performance do algoritmo 
de aprendizagem de máquina, logo selecionar de maneira apropriada os atributos de entrada auxilia no desempenho da classificação.

O código abaixo consiste na Seleção de Atributos de determinado Banco de Dados baseado na similaridade de cossenos 
entre 2 atributos. É utilizado a Otimização por Colônia de Formigas para avaliar subsets através de uma modelagem 

■ Keywords: Feature Selection; Ant Colony Optimization; Dimensionality Reduction; IBM SPSS Modeler; Data Classification; Wrapper Methods; Python.
  
# Algoritmos de referência (indicados na pasta)

* Ant Colony Optimization aplicado ao problema do caixeiro viajante (Traveling Salesman Problem - TSP).
  Link de código [link](https://github.com/marcoscastro/tsp_aco)
  
* Ant_Colony_Optimization_Feature_Selection.py
  Link de código [link](https://github.com/sssalam1/Optimization-Codes/blob/master/Ant_Colony_Optimization_Feature_Selection.py)
  
* UFSACO: Unsupervised Feature Selection using Ant Colony Optimization
Simulation of an Unsupervised Feature Selection using Ant Colony Optimization (UFSACO) algorithm. System is implemented in Python 2.7.11.
Link for algorithm details: [Paper](https://https://www.researchgate.net/publication/261371258_An_unsupervised_feature_selection_algorithm_based_on_ant_colony_optimization) 
(Segue pasta artigos)

* WFACOFS: Wrapper Filter based Ant Colony Optmization for Feature Selection
A wrapper-filter feature selection technique based on ant colony optimization
Link for algorithm details: [Paper](https://link.springer.com/article/10.1007/s00521-019-04171-3)(Segue pasta artigos)
Link de código em .m [link](https://github.com/ManosijGhosh/Feature-Selection-Algorithm/tree/master/WFACOFS)

# Proposta(em desenvolvimento):
FS_ACO
Projetado para selecionar atributos de determinado BD através da ACO adotando a similaridade de cosseno entre pares de atributos como peso. A performance dos subsets(acuracia) através de uma modelagem será avaliada e ao final apresentado o conjunto de maior valor. A atualização do feromônio e regra de probabilidade foram desenvolvidas conforme algoritmo Ant-System
