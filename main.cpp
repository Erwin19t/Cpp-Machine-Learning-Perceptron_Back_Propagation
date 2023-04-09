/*
Programa de Perceptron V1

Compilacion: g++ main.cpp -o Ejecutable -std=c++11
Ejecucion: ./Ejecutable
*/

#include <iostream>
#include <unistd.h>
#include <cmath>
#define Alpha 0.005

using namespace std;

double  Initial_Weights(int);
double  Evaluate(double*, double*, int, int, int);
double  Sigmoid(double);
int     Threshold(double);
double  D_Sigmoid(double);
double* Gradient(double*, double*, double, int, int, int);
void    W_Update(double*, double*, double*, double, int, int);
void    Train(double*, double*, double*,  int, int, int);
void    PrintAll(int, int, int, double, double, double, double*, double*);
void    Test(double*, double*, int, int);

int main(){
    double X[]  = {1, 1, 1, 1,  //Bias
                   0, 0, 1, 1,  //X1
                   0, 1, 0, 1}; //X2

    double Y_and[] = {0, 0, 0, 1};
    double Y_or[]  = {0, 1, 1, 1};
    double Y_xor[] = {0, 1, 1, 0};
    //Omega debe ser un vector columna, pero 
    //para simplificar la programacion se considerara
    //vector fila.
    double w_and[] = {1 , 1, 1};
    double w_or[]  = {1 , 1, 1};
    double w_xor[] = {1 , 1, 1};

    int n_rows   = 3;
    int n_cols   = 4;
    int Max_Iter = 100000;

    for(int i = 0 ; i < n_rows ; i++){
        w_and[i] = Initial_Weights(i+1);
        w_or[i]  = Initial_Weights(i+2);
        w_xor[i] = Initial_Weights(i+3);
    }
    cout<<"Pesos Iniciales"<<endl;
    cout<<"W_AND = ["<<w_and[0]<<" , "<<w_and[1]<<" , "<<w_and[2]<<"]"<<endl;
    cout<<"W_OR  = ["<< w_or[0]<<" , "<< w_or[1]<<" , "<< w_or[2]<<"]"<<endl;
    cout<<"W_XOR = ["<<w_xor[0]<<" , "<<w_xor[1]<<" , "<<w_xor[2]<<"]"<<endl<<endl;

    Train(X, w_and, Y_and, n_cols, n_rows, Max_Iter);
    Train(X, w_or , Y_or , n_cols, n_rows, Max_Iter);
    Train(X, w_xor, Y_xor, n_cols, n_rows, Max_Iter);

    cout<<"Pesos Finales"<<endl;
    cout<<"W_AND = ["<<w_and[0]<<" , "<<w_and[1]<<" , "<<w_and[2]<<"]"<<endl;
    cout<<"W_OR  = ["<< w_or[0]<<" , "<< w_or[1]<<" , "<< w_or[2]<<"]"<<endl;
    cout<<"W_XOR = ["<<w_xor[0]<<" , "<<w_xor[1]<<" , "<<w_xor[2]<<"]"<<endl<<endl;

    cout<<"Test de Pesos"<<endl;
    cout<<"Compuerta AND"<<endl;
    Test(X, w_and, n_cols, n_rows);
    cout<<endl<<endl;
    
    cout<<"Compuerta OR"<<endl;
    Test(X, w_or, n_cols, n_rows);
    cout<<endl<<endl;
    cout<<"Compuerta XOR"<<endl;
    Test(X, w_xor, n_cols, n_rows);
    
    return 0;
}

//Paso 1: Inicializar pesos aleatoriamente
double Initial_Weights(int id){
    double Theta;
    int Sign;
    srand(id + (int)time(NULL));
    Sign = 1 + rand()%100;
    if(Sign%2 == 0){    //Para genera numeros pseudoaleatorios positivos
        srand(id + (int)time(NULL));
        //Theta = 1 + rand()%9;
        Theta = 1 + ((rand()%(2001))/1000.00f);
    }
    else{               //Para generar numeros pseudoaleatorios negativos
        srand(id + (int)time(NULL));
        Theta = -1 + ((rand()%(-2001))/1000.00f);
    }
    return Theta;
}

//Argumento de la Sigmoidal
double Evaluate(double* x, double* omega, int ncols, int col, int nrows){
    double z = 0.0;
    for(int i = 0 ; i < nrows ; i++){
        z += x[i*ncols + col] * omega[i];
    }
    return z;
}

//Sigmoidal
double Sigmoid(double z){
    return 1 / (1 + exp(-z));
}

//Umbralizacion
int Threshold(double Y_g){
    if(Y_g >= 0.5)
        return 1;
    return 0;
}

//Derivada de Sigmoidal
double D_Sigmoid(double z){
    return Sigmoid(z) * (1 - Sigmoid(z));
}

//Calculo del Vector Gradiente
double* Gradient(double* x, double* omega, double z, int ncols, int col, int nrows){
    double* Grad = new double [nrows];
    for(int i = 0 ; i < nrows ; i++){
        Grad[i] = D_Sigmoid(z) * x[i*ncols + col];
    }
    return Grad;
}

//Actualizacion de pesos
void W_Update(double* omega, double* Gradient, double* Y, double z, int nrows, int col){
    double sigmoid = Sigmoid(z);
    int Y_g = Threshold(sigmoid);
    for(int i = 0 ; i < nrows ; i ++){
        if((Y_g == 1) && (Y[col] == 0)){
            omega[i] -= Alpha * Gradient[i]; 
        }
        if((Y_g == 0) && (Y[col] == 1)){
            omega[i] += Alpha * Gradient[i]; 
        }
    }
}

//Funcion de Entrenamiento
void Train(double* x, double* omega, double* Y, int ncols, int nrows, int maxiter){
    int Counter  = 0;
    int col  = 0;

    while(Counter < maxiter){
        double Z = Evaluate(x, omega, ncols, col, nrows);
        double sigmoid = Sigmoid(Z);
        int Y_g = Threshold(sigmoid);
        double d_sigmoid = D_Sigmoid(Z);
        double* gradient = Gradient(x, omega, Z, ncols, col, nrows);
        W_Update(omega, gradient, Y, Z, nrows, col);

        //PrintAll(Counter, col, Y_g, Z, sigmoid, d_sigmoid, gradient, omega);

        if(col < 3){
            col++;
        }
        else{
            col = 0;
            Counter++;
        }
    }
}

void Test(double* x, double* omega, int ncols, int nrows){
    double* Evaluation = new double [ncols];
    double* Y_g = new double [ncols];
    for(int i = 0 ; i < ncols ; i++){
        Evaluation[i] = Evaluate(x, omega, ncols, i, nrows);
        Y_g[i] = Threshold(Sigmoid(Evaluation[i]));
    }
    cout<<"x_1 , x_2 , y"<<endl;
    cout<<" "<<x[4]<<"  ,  "<<x[8]<<"  , "<<Y_g[0]<<endl;
    cout<<" "<<x[5]<<"  ,  "<<x[9]<<"  , "<<Y_g[1]<<endl;
    cout<<" "<<x[6]<<"  ,  "<<x[10]<<"  , "<<Y_g[2]<<endl;
    cout<<" "<<x[7]<<"  ,  "<<x[11]<<"  , "<<Y_g[3]<<endl;
}

void PrintAll(int Counter, int col, int Y_g, double Z, double sigmoid, double d_sigmoid, double* gradient, double* omega){
    cout<<"Iteracion "<<Counter+1<<" , Columna: "<<col+1<<endl;
    cout<<"La evaluacion es: "<<Z<<endl;
    cout<<"La sigmoidal es:"<<sigmoid<<endl;
    cout<<"Y_g es: "<<Y_g<<endl;
    cout<<"La derivada de sigmoidal es: "<<d_sigmoid<<endl;
    cout<<"El gradiente es:"<<endl;
    cout<<"G[0] = "<<gradient[0]<<endl;
    cout<<"G[1] = "<<gradient[1]<<endl;
    cout<<"G[2] = "<<gradient[2]<<endl;
    cout<<"El vector de pesos es:"<<endl;
    cout<<"w[0] = "<<omega[0]<<endl;
    cout<<"w[1] = "<<omega[1]<<endl;
    cout<<"w[2] = "<<omega[2]<<endl<<endl;
}