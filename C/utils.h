#ifndef UTILS_HH
#define UTILS_HH

#include <gsl/gsl_matrix.h>

double randfrom(double min, double max);

void setRandomMat(gsl_matrix* mat, double min, double max);

void printMatrix(gsl_matrix* mat);

void multiplyMatrix(gsl_matrix* a, const gsl_matrix* b);

void fromArrayToColumn(gsl matrix*a, const double vec[]);

#endif
