#include "iostream"
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

using namespace std;

typedef map<string, map<int, double>> map_bag_words;
typedef unordered_map<string, double> umap;
typedef vector<pair<string, double>> vector_sd;

void leer_preprocesado(vector<umap> &a_documentos) {
    ifstream infile("pre_eric_ds.csv");
    string line;
    string buffer;
    string word;
    double val;
    while (std::getline(infile, line))
    {
        line = line.substr(1, line.size()-2);
        umap a_caracteristicas;
        stringstream ss(line);
        while(getline(ss, buffer, ',')) {
            word = buffer.substr(3, buffer.size()-4);
            getline(ss, buffer, ',');
            val = stod(buffer.substr(0, buffer.size()-1));
            a_caracteristicas[word] = val;        
        }

        a_documentos.push_back(a_caracteristicas);

    }
}

void pesos(double** &mat_pesos, umap a_caracteristicas) {
    int i = 0;
    int j = 0;
    for(auto caracteristica_i : a_caracteristicas) {
        j =0;
        for(auto caracteristica_j : a_caracteristicas) {
            mat_pesos[i][j] = abs(caracteristica_i.second - caracteristica_j.second);
            if (mat_pesos[i][j] == 0) mat_pesos[i][j] = 0.0001;
            ++j;
        }
        ++i;
    }
}

void visibilidad(double** &mat_visibilidad, double** &mat_pesos, int size) {
    for(int i=0; i< size; ++i) {
        for(int j=0; j< size; ++j) {
            if(mat_pesos[i][j] != 0.0)
                mat_visibilidad[i][j] = 1./mat_pesos[i][j];
        }
    }
}

void hormiga_aleatoria(int* &a_pos_hor,int num_hor,int num_car) {
    srand (time(NULL));
    for (int i=0; i<num_hor; ++i) {
        a_pos_hor[i] = rand() % num_car;
    }
}

void visitar(int** &matriz, int i, int j) {
    matriz[i][j] += 1;
}

bool regla_transicion(double q_ini) {
    double q = (double) rand() / (RAND_MAX);
    if ( q <= q_ini) {
        return 0;
    } else {
        return 1;
    }
}



int seleccion_avariciosa(int** &mat_visitados, int h, double* &a_taos,double** &mat_visibilidad, int pos_hormiga, double beta, int num_caracteristicas) {
    double mayor = 0.0;
    int pos_mayor = 0;
    for (int i=0; i<num_caracteristicas; ++i) {
        if (mat_visitados[h][i] == 0) {
            double tao_vis = a_taos[i] * pow(mat_visibilidad[pos_hormiga][i], beta);
            if(tao_vis > mayor) {
                mayor = tao_vis;
                pos_mayor = i;
            }
        }
    }
    return pos_mayor;
}

int seleccion_probabilistica(int** &mat_visitados, int h, double* &a_taos,double** &mat_visibilidad, int pos_hormiga, double beta, int num_caracteristicas) {

    double sumatoria = 0.0;
    for (int i=0; i<num_caracteristicas; ++i) {
        if (mat_visitados[h][i] == 0) {
            sumatoria += ( a_taos[i] * pow(mat_visibilidad[pos_hormiga][i], beta) );
        }
    }

    double j;
    double n_random = (double) rand() / (RAND_MAX);
    double sum_esc = 0.0;
    for (int i=0; i<num_caracteristicas; ++i) {
        if (mat_visitados[h][i] == 0) {
            j = a_taos[i] * pow(mat_visibilidad[pos_hormiga][i], beta) / sumatoria;
            sum_esc += j;
            if (sum_esc > n_random) {
                return i;
            }
        }
    }
}

int main()
{

    int num_ciclos = 10; //numero de ciclos
    int num_hormigas = 8 ;
    double tao = 0.2;
    int num_caracteristicas_final = 5;
    int num_caracteristicas_limite = 50;
    double q_ini = 0.7;
    double beta = 0.35;
    double evaporacion=0.2;
    
    vector<umap> a_documentos;
    leer_preprocesado(a_documentos);


    double** mat_pesos;
    double** mat_visibilidad;
    int* list_cont_car; // suma de la matriz mat_cont_car
    int** mat_cont_car; // resultados de las hormigas
    int** mat_visitados; // matriz de visitados
    double* a_taos;
    int* a_pos_hor;

    // BEGIN INICIO TIEMPO
    time_t rawtime;
    struct tm * timeinfo;
    char buffer_time[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer_time,sizeof(buffer_time),"%d-%m-%Y_%H:%M:%S",timeinfo);
    string str_time(buffer_time);
    // END INICIO TIEMPO

    ofstream outfile("documentos_"+str_time+".csv");

    map_bag_words bag_of_words;

    int num_cont_documentos = 0;
    vector<umap>::iterator it = a_documentos.begin();
    for ( ; it != a_documentos.end(); ++it) {
        int num_caracteristicas = (*it).size();
        if (num_caracteristicas == 0) {
            outfile << endl;
            ++num_cont_documentos;
            continue;
        }
        
        if (num_caracteristicas < num_caracteristicas_final) {
            outfile << endl;
            ++num_cont_documentos;
            continue;
        }

        umap a_caracteristicas;

        // obtenemos las caracteristicas mas importantes
        if (num_caracteristicas > num_caracteristicas_limite) {
            vector_sd vector_caracteristicas_tmp;
            for (auto caracteristica : (*it)) {
                pair<string,double> a(caracteristica.first, caracteristica.second);
                vector_caracteristicas_tmp.push_back(a);
            }
            sort(begin(vector_caracteristicas_tmp), end(vector_caracteristicas_tmp), [](auto p1, auto p2){return p1.second > p2.second;});
            int i = 0;
            for (auto caracteristica : vector_caracteristicas_tmp) {
                if (i>=num_caracteristicas_limite) break;
                a_caracteristicas[caracteristica.first] = caracteristica.second;
                ++i;
            }
            num_caracteristicas = num_caracteristicas_limite;
        } else {
            a_caracteristicas = (*it);
        }

        mat_pesos = new double *[num_caracteristicas];
        mat_visibilidad = new double *[num_caracteristicas];
        list_cont_car = new int [num_caracteristicas];
        a_taos = new double [num_caracteristicas];
        for(int i=0; i<num_caracteristicas; ++i) {
            mat_pesos[i] = new double[num_caracteristicas];
            mat_visibilidad[i] = new double[num_caracteristicas];
            list_cont_car[i] = 0;
        }
        // hormigas
        mat_cont_car = new int *[num_hormigas];
        mat_visitados = new int *[num_hormigas];
        a_pos_hor = new int [num_hormigas];
        for (int i=0; i<num_hormigas; ++i) {
            mat_cont_car[i] = new int[num_caracteristicas];
            mat_visitados[i] = new int[num_caracteristicas];
            for (int j=0; j<num_caracteristicas; ++j) {
                mat_cont_car[i][j] = 0;
                mat_visitados[i][j] = 0;
            }
        }

        pesos(mat_pesos, a_caracteristicas);
        visibilidad(mat_visibilidad, mat_pesos, num_caracteristicas);

        // generamos el array de taos(feromona) del tamano de las caracteristicas
        for (int i=0; i<num_caracteristicas; ++i) {
            a_taos[i] = tao;
        }

        for (int nCiclo=0; nCiclo < num_ciclos; ++nCiclo){
            for (int i=0; i<num_caracteristicas; ++i) {
                list_cont_car[i] = 0;
            }
            for (int i=0; i<num_hormigas; ++i) {
                for (int j=0; j<num_caracteristicas; ++j) {
                    mat_cont_car[i][j] = 0;
                    // array de visitados x cada hormiga
                    //     A  B  C
                    // H1 [0, 0, 0]
                    // H2 [0, 0, 0]
                    // H3 [0, 0, 0]
                    mat_visitados[i][j] = 0;
                }
            }
            // generamos array de posiciones aleatorias de las hormigas(3)
            // [1,1,2]
            hormiga_aleatoria(a_pos_hor, num_hormigas, num_caracteristicas);
            // ubicar a las hormigas en el grafo
            // [0 , 0, 1] 2
            // [0 , 1, 0] 1
            for (int h=0; h<num_hormigas; ++h) {
                visitar(mat_visitados, h, a_pos_hor[h]);
            }
            
            for(int i=0; i<num_hormigas; ++i) {
                list_cont_car[a_pos_hor[i]] += 1;
            }


            for(int i=0; i<num_hormigas; ++i) {
                for(int j=0; j<15-1; ++j) { 
                    // posición de la hormiga
		            int pos_hormiga = a_pos_hor[i];

                    // escoger la siguiente caracteristica, avariciosa o probabilistica
                    int next_pos;
                    if (regla_transicion(q_ini) == false) {
                        next_pos = seleccion_avariciosa(mat_visitados, i, a_taos, mat_visibilidad, pos_hormiga, beta, num_caracteristicas);
                    } else {
                        next_pos = seleccion_probabilistica(mat_visitados, i, a_taos, mat_visibilidad, pos_hormiga, beta, num_caracteristicas);
                    }

                    // actualizamos array de visitados de la hormiga y la posición 
		            // de la caracteristica escogida
                    visitar(mat_visitados, i, next_pos);

                    // movemos la hormiga a la nueva característica seleccionada
                    a_pos_hor[i] = next_pos;

                    // actualizamos contador de la característica seleccionada
		            mat_cont_car[i][next_pos] += 1;
                }
            }

            for(int j=0; j<num_caracteristicas; ++j) {
                for(int i=0; i<num_hormigas; ++i) {
                    list_cont_car[j] += mat_cont_car[i][j];
                }
            }
            
            // actualización global del tao(feromona)
            int sumatoria_fc = 0;
            double division_fc = 0.0;
            for (int i=0; i<num_caracteristicas; ++i) {
                sumatoria_fc += list_cont_car[i];
            }
            for (int i=0; i<num_caracteristicas; ++i) {
                if (sumatoria_fc==0) {
                    division_fc = 0.0;
                } else {
                    division_fc = (double)list_cont_car[i] / (double)sumatoria_fc;
                }
                a_taos[i] = (1-evaporacion) * a_taos[i] + division_fc;
            }
        }

        // obtenemos las "num_caracteristicas_final" principales caracteristica
        vector_sd vector_feromonas;
        int i = 0;
        for (auto caracteristica : a_caracteristicas) {
            pair<string,double> a(caracteristica.first, a_taos[i]);
            vector_feromonas.push_back(a);
            ++i;
        }
        sort(begin(vector_feromonas), end(vector_feromonas), [](auto p1, auto p2){return p1.second > p2.second;});


        i = 0;
        for (auto caracteristica : vector_feromonas) {
            if (i>=num_caracteristicas_final) continue;
            ++i;
            
            outfile << caracteristica.first << "," << caracteristica.second << ",";
            bag_of_words[caracteristica.first][num_cont_documentos] = caracteristica.second;
        }
        outfile << endl;

        cout << "documento #" << num_cont_documentos << ", " << num_caracteristicas << " palabras" << endl;

        ++num_cont_documentos;

        // eliminar matriz de pesos
        // eliminar matriz de visibilidad
        for(int i=0; i<num_caracteristicas; ++i) {
            delete[] mat_pesos[i];
            delete[] mat_visibilidad[i];
        }
        delete[] mat_pesos;
        delete[] mat_visibilidad;
        // eliminar list_cont_car
        delete[] list_cont_car;
        // eliminar mat_cont_car
        // eliminar mat_visitados
        for (int i=0; i<num_hormigas; ++i) {
            delete[] mat_cont_car[i];
            delete[] mat_visitados[i];
        }
        delete[] mat_cont_car;
        delete[] mat_visitados;
        // eliminar a_taos
        delete[] a_taos;
        // eliminar a_pos_hor
        delete[] a_pos_hor;
    }

    // BEGIN FIN TIEMPO
    time_t rawtimeend;
    struct tm * timeinfoend;
    char buffer_time_end[80];
    time (&rawtimeend);
    timeinfoend = localtime(&rawtimeend);
    strftime(buffer_time_end,sizeof(buffer_time_end),"%d-%m-%Y_%H:%M:%S",timeinfoend);
    string str_time_end(buffer_time_end);
    // END FIN TIEMPO

    
    double timediff = difftime(rawtimeend,rawtime);
    double secondsdiff = (int)timediff % 60;
    double minutesdiff = (int)timediff / 60;

    cout << "Inicio    : " << str_time << endl;
    cout << "Finalizo  : " << str_time_end << endl;
    cout << "Total     : " << timediff << " s" << endl;
    cout << "Demoro    : " << minutesdiff << " minutos, " << secondsdiff << " segundos" << endl;

    cout << endl << "Escribiendo matriz" << endl;

    return 0;
}
