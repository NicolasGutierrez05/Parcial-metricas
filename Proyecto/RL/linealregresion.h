#ifndef LINEALREGRESION_H
#define LINEALREGRESION_H
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

class LinealRegresion
{
public:
    LinealRegresion(){}
    float FunCostoOLS(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradDesc(Eigen::MatrixXd X,
                                                   Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha,
                                                   int iteraciones);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // LINEALREGRESION_H
