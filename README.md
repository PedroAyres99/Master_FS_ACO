# SELECÃO DE ATRIBUTOS BASEADO NO ALGORITMO DE OTIMIZACÃO COLÔNIA DE FORMIGAS PARA PROCESSOS MINERADORES

Repositório referente à pesquisa entitulada acima. Programa: Mestrado Profissional em Instrumentação, Controle e Automação de Processos de Mineração - PROFICAM. Instituições: UFOP / Instituto Tecnológico VALE - ITV.

Feature Selection (Seleção de Atributos): consiste em algoritmos com propósitos de reduzir a dimensionalidade de banco de dados, 
selecionando sub-conjuntos relevantes para a construção de modelos preditivos. Atributos redundantes prejudicam a performance do algoritmo 
de aprendizagem de máquina, logo selecionar de maneira apropriada os atributos de entrada auxilia no desempenho da classificação.

■ Keywords: Feature Selection; Ant Colony Optimization; Dimensionality Reduction; IBM SPSS Modeler; Data Classification; Wrapper Methods; Python.
  
# Algoritmos de referência (indicados na pasta)

* Ant Colony Optimization aplicado ao problema do caixeiro viajante (Traveling Salesman Problem - TSP).
  Link de código [link](https://github.com/marcoscastro/tsp_aco)
  
* Ant_Colony_Optimization_Feature_Selection.py
  Link de código [link](https://github.com/sssalam1/Optimization-Codes/blob/master/Ant_Colony_Optimization_Feature_Selection.py)
  
* UFSACO: Unsupervised Feature Selection using Ant Colony Optimization / 
Simulation of an Unsupervised Feature Selection using Ant Colony Optimization (UFSACO) algorithm. System is implemented in Python 2.7.11.
Link for algorithm details: [Paper](https://https://www.researchgate.net/publication/261371258_An_unsupervised_feature_selection_algorithm_based_on_ant_colony_optimization) 
(Segue pasta artigos)

* WFACOFS: Wrapper Filter based Ant Colony Optmization for Feature Selection
A wrapper-filter feature selection technique based on ant colony optimization
Link for algorithm details: [Paper](https://link.springer.com/article/10.1007/s00521-019-04171-3)(Segue pasta artigos)
Link de código em .m [link](https://github.com/ManosijGhosh/Feature-Selection-Algorithm/tree/master/WFACOFS)

# Proposta:

  O método proposto neste projeto baseia-se no desenvolvimento de um algoritmo de seleção de atributos do tipo Filter-Wrapper baseado na ACO, observando-se a tendência de recentes literaturas propostas na área. Foi adotado como estrutura de código para a otimização por colônia de formigas o proposto no algoritmo Ant Systems e a partir dele realizado as alterações de forma que o método apresente características do tipo \textit{Filter} e \textit{Wrapper}. O algoritmo foi desenvolvido adotando-se a linguagem Python 3.8 com o auxílio de ferramentas disponibilizadas pelas bibliotecas relacionadas à Preparação de Dados, Estatísticas e Machine Learning. É adotado como função de avaliação a métrica Acurácia.
 
O fluxograma do algoritmo é descrito através da figura abaixo:

![Fluxo_FS_ACO_3](https://user-images.githubusercontent.com/53266990/138096556-7b2b161b-5729-46dd-9a38-c351f5981e85.png)
