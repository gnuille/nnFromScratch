#ifndef UTILS_HH
#define UTILS_HH

#include <gsl/gsl_matrix.h>
#include <math.h>
#include <assert.h>

typedef double (*funMat)(double inp);

double randfrom(double min, double max);

int randint(int n);

void setRandomMat(gsl_matrix* mat, double min, double max);

void printMatrix(const gsl_matrix* mat);

void multiplyMatrix(const gsl_matrix* a,const gsl_matrix* b, gsl_matrix* ret);

void fromArrayToColumn(gsl_matrix*a, const double vec[]);

void fromArrayToLine(gsl_matrix*a, const double vec[]);

void actSigmoid(gsl_matrix* inp);

void applyFunMatrix(gsl_matrix* inp, funMat fun);

double sigmoid(double d);

double sigmoidDerivate(double d);

double relu(double d);

double reluDerivative(double d);

void printMatrixArray(gsl_matrix** mats, int size);

double* getRowAsArray(gsl_matrix* mat, int row);

double absoluteValue(double d);

#endif
