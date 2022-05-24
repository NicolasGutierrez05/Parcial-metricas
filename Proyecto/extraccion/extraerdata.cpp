#include "extraerdata.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <stdlib.h>
#include <string>
#include <fstream>
/************************************************
 * Primer funcion miembro: lectura fichero CSV
 * vector de vectores "string". La idea es leer
 * linea por linea y almacenar en un vector
 * de vectores del tipo "String".
 * **********************************************/

std::vector<std::vector<std::string>> ExtraerData::ReadCSV(){
    // Se abre el fichero solo para lectura
    std::ifstream Fichero(setDatos);
    // Vector de vectores "string: tendrá los datos del dataset
    std::vector<std::vector<std::string>> datosString;
    /* Se itera a traves de cadda linea del dataset,
     * al tiempo que se divide la linea, con el delimitador
     */
     std::string linea = "";
     while(getline(Fichero,linea)){
         std::vector<std::string> vectorFila;
         // dividimos segun el delimitador
         boost::algorithm::split(vectorFila,linea,boost::is_any_of(delimitador));
         datosString.push_back(vectorFila);
     }
    /* Se cierra el fichero*/
     Fichero.close();
     // Se retorna el vector de vectores String
     return datosString;

}


/* Segunda funcion para guardar el vector de vectores
 * del tipo string: se almacena en una matriz
 * para presentar como "un bjeto parecido al DATAFRAME
 * ue entrega PANDAS." */
Eigen::MatrixXd ExtraerData::CSVtoEigen(
        std::vector<std::vector<std::string>> setDatos,
        int filas, int columnas){
    // Si tiene cabecera se remueve, es decir, solo se manipulan
    // los datos
    if( header == true){
        filas = filas - 1;
    }
    /* se itera sobre filas y columnas para almacenar en la matrix de tamaño
     * filasxcolumnas.
     * BAsicamente se alamacenará "strings" en el vector :
     * luego se pasan a float para ser manipulados*/

    //se crea la matrix vacia
    Eigen::MatrixXd dfMatrix(columnas,filas);

    for(int i = 0; i< filas;i++){
        for(int j = 0; j<columnas;j++){
            dfMatrix(j,i) = atof(setDatos[i][j].c_str());// ocn atof se pasa al tipo float los string
        }
    }
    /* Se transpone la matriz para que sea filas x columnas se devuelve o retonra la matriz */
    return dfMatrix.transpose();

}

/*Se hace la función que retorne el promedio por cada dato
 * (columna). La idea es comparar con l hecho en python - pandas- sklearn
 * para verificar que la función artesanl corresponda (validad) */

/* Auto y decltype: especifica el tipo de variableque se empieza a declarar,
 * la cual la deducirá de forma automática con su inicializador (tiempo de compilación).
 * Para las funciones, si el tipo de retorno es un "auto", se evaluará mediante la expresión
 * del tipo de retorno en tiempo de compilación */

auto ExtraerData::Promedio(Eigen::MatrixXd datos) ->
decltype (datos.colwise().mean()){
    return datos.colwise().mean();
}

/* Función Desviación Standart:
 * data = xi - x.promedio
 */

auto ExtraerData::DesvStand(Eigen::MatrixXd data) ->
decltype (((data.array().square().colwise().sum()) / (data.rows()-1)).sqrt()){
    return ((data.array().square().colwise().sum()) / (data.rows()-1)).sqrt();
}

/* Acto seguido se procede a hacer el cálculo o la función de normalización:
la idea es evitar los cambios bruscos en los datos ( cambios en orden de magnitud).
Lo anterior representa un deteriior para la predicción, sobre la base de cualquier módelo
de machine learnign. (Evita los outliers) */

Eigen::MatrixXd ExtraerData::Normalizador(Eigen::MatrixXd datos){
    // Normalización
    // MatrixNorm = ( xi -x.mean() ) / Desviación estandar

    /* Primero se extrae-calcula el promedio */
    //auto PromedioD = Promedio(datos);
    // Se realiza la diferencia ( datos escalados = dato - promedio)
    Eigen::MatrixXd DataEscalado = datos.rowwise() - Promedio(datos);
    // Segundo se calcula la desviación
    // auto DesviacionD = DesvStand(DataEscalado);

    // Se retorna cada dato escalado
    Eigen::MatrixXd MatrixNorm = DataEscalado.array().rowwise() / DesvStand(DataEscalado);
     return MatrixNorm;

}

/* A CONTINUACION se implementa la funcion para la division de los datos:
 entrenamiento y prueba.
 La idea es crear 4 matrices que tengan los datos de la variables
 dependientes e independientes para el entrenamiento y las pruebas.
 Similar a la función con sklearn que seria: train_test_split()
*/

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
ExtraerData::TrainTestSplit(Eigen::MatrixXd dataNorm, float sizeTrain){
    /* Numero de filas */
    int filas = dataNorm.rows();
    /* Numero de filas para entrenamiento: */
    int filasTrain = round (sizeTrain * filas) ;
    /* Numero de filas para prueba*/
    int filasTest = filas - filasTrain;

    /* Con Eigen se puede especificar un  bloque de una matriz superior
     * o inferior a partir de la fila que quieras como final del bloque o
     * como principio del bloque; para este caso en especial se seleccionará
     * como entrenamiento el bloque superior de la matriz dataNorm.
     * Se deja entonces la matriz inferior para prueba.
     */

     Eigen::MatrixXd Train = dataNorm.topRows(filasTrain);

      std::cout<<filasTest<<std::endl;

     /* Para este caso en especial (dataset de entrada) se tiene que
      * los datos de las columnas se identifican en la parte izquierda
      * las features o variables independientes quedando la primera columna
      * de la derecha como variable dependiente.
      */

     /* Se crea una matriz correspondiente a las features o variables
      * independientes.
      */

     Eigen::MatrixXd X_train = Train.leftCols(dataNorm.cols() - 1);

     /* Se crea una matriz correspondiente a la variable dependiente
      * a la primera columna.
      */

     Eigen::MatrixXd y_train = Train.rightCols(1);

     /* Se hace el mismo procedimiento para prueba
      */

     Eigen::MatrixXd Test = dataNorm.bottomRows(filasTest);
     Eigen::MatrixXd X_test = Test.leftCols(dataNorm.cols()-1);
     Eigen::MatrixXd y_test = Test.rightCols(1);

     return std::make_tuple(X_train,y_train,X_test,y_test);

}
/* A continuación se desarrollan dos ufnciones para la manipulación de vectores y la conversión
 * fichero a matriz Eigen. La manipulación de vectores representa iterar por el fichero de
 * entrada y convertirlo en vector flotante. */
void ExtraerData::conVectorFichero(std::vector<float> vectorDatos, std::string fileName ){
    /* Se crea un objeto que tendrá la lectura del fichero. */
    std::ofstream ficheroSalida(fileName);
    /* Se itera sobre el fichero de salida con el delimitador cambio de linea (\n),
     * para ser copiado en un vector (vectorSalida) */
    std::ostream_iterator<float> salidaIterador(ficheroSalida, "\n");

    /* Se copian los elementos del iterador en el vector de datos. */
    std::copy(vectorDatos.begin(), vectorDatos.end(),salidaIterador);

}

/* A continuación se desarrolla la función de conversión de matriz Eigen a
 * fichero. Función útil dado que los valores parciales que se obtienen
 * se imprimen en ficheros para tener seguridad y trazabilidad. */
void ExtraerData::EigenToFile(Eigen::MatrixXd matrixData, std::string fileName){
    /* Se crea un objeto que tendrá la lectura del fichero. */
    std::ofstream ficheroSalida(fileName);
    if(ficheroSalida.is_open()){
        ficheroSalida << matrixData << "\n";
    }
}
