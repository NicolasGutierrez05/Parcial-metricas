#ifndef EXTRAERDATA_H
#define EXTRAERDATA_H
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

/* En la clse se implementa el constructor
 * que recibe los argumentos d eentrada de la clase*/

class ExtraerData
{
    // Recibe el nombre del fichero CSV
    std::string setDatos;
    //recibe el separador o delimitador
    std::string delimitador;
    //recibe si tiene o no caBECERA EL FICHERO DE DATOS
    bool header;

public:
    ExtraerData(std::string datos, std::string separador, bool head):
        setDatos(datos),
        delimitador(separador),
        header(head) {}
    /* Prototipo de funciones*/
    std::vector<std::vector<std::string>> ReadCSV();
    Eigen::MatrixXd CSVtoEigen(
            std::vector<std::vector<std::string>> setDatos,
            int filas, int columnas);

    auto Promedio(Eigen::MatrixXd datos) ->
    decltype (datos.colwise().mean());

    auto DesvStand(Eigen::MatrixXd data) ->
    decltype (((data.array().square().colwise().sum()) / (data.rows()-1)).sqrt());

    Eigen::MatrixXd Normalizador(Eigen::MatrixXd datos);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
    TrainTestSplit(Eigen::MatrixXd dataNorm, float sizeTrain);
    void conVectorFichero(std::vector<float> vectorDatos, std::string fileName );
    void EigenToFile(Eigen::MatrixXd matrixData, std::string fileName);

};

#endif // EXTRAERDATA_H
