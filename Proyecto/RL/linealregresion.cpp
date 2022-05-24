#include "linealregresion.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>
#include <string>
#include <fstream>

/* En esta clase se desarrolla la función OLS y el gradiente
 * descendiente tal y como ha sido demostrado en clase. */

/* Se necesita entrenar el modelo, lo que implica minimizar alguna
 * función de costo (Se selecciona OLS), la idea es medir la precisión
 * de la función de  hipótesis. La función de costo es la forma de penalizar
 * al modelo  por cometer un error. Se implementa una función que retorna un
 * flotante, que toma como entrada el dataset (X,y), debe retornar junto con los
 * coeficientes (m1, m2, ... mn, b) Dibujará un hiperplano. */


float LinealRegresion::FunCostoOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta)
{
    Eigen::MatrixXd diferencia = pow((X*theta-y).array(),2);
    return (diferencia.sum()/(2*X.rows()));
}

/* Se necesita proveer al programa una función para dar al
 * algoritmo un valor inicial para theta, elc uam cambiara
 * iterativamente hasta que converja el valor al m´inimo de
 * nuestra función de costo. Básicamente describe el gradiente
 * descendiente: La idea es calcular el gradiente para nuestra
 * función de costo dada por la derivada parcial de la función.
 * La función tendrá un apha que representa el salto del
 * gradiente. Las entradas para la función sera X, y, theta y
 * el número de veces para acualizar theta hasta que la funcion
 * converja. */

std::tuple<Eigen::VectorXd, std::vector<float>> LinealRegresion::GradDesc(Eigen::MatrixXd X,
                                               Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha,
                                               int iteraciones){
    /* Se almacena temporalmente los parámetros theta. */
    Eigen::MatrixXd temporalTheta = theta;
    /* Se extrae la cantidad de parámetros (m: features). */
    int parametros = theta.rows();
    /* Se ubica el costo inicial, que se actualiza con cada paso y
     * los nuevos pesos. */
    std::vector<float> costo;
    costo.push_back(FunCostoOLS(X,y,theta));

    /* Por cada iteración se calcula la función de error, que se usa
     * para multiplicar cada dimensión o feature y así almacenarlo en
     * la variable temporal. Se actualiza theta y se calcula el nuevo
     * valor de la función de costo basada en el nuevo valor de theta. */
    for(int i =0; i<iteraciones; i++){
        Eigen::MatrixXd error = X*theta-y;
        for(int j=0; j<parametros; j++){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd valorTemp = error.cwiseProduct(X_i);
            temporalTheta(j,0) = theta(j,0)-((alpha/X.rows())*valorTemp.sum());
        }
        theta = temporalTheta;
        costo.push_back(FunCostoOLS(X,y,theta));
    }
 /* se empaqueta la tupla para ser entregada. */
    return std::make_tuple(theta,costo);

}

/* Para determinar que tan bueno es el modelo que se ha desarrollado a continuación
se implementa una función como metrica de evaluación: R2. R2 representa una medida
de que tan bueno es el modelo. y_hat son las predicciones.
La funcion va a retornar un flotante */
float LinealRegresion::RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y-y_hat).array(), 2).sum();
    auto denominador = pow(y.array()-y.mean(), 2).sum();

    return 1-(numerador/denominador);
}
