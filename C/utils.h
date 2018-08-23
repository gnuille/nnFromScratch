#ifndef UTILS_HH
#define UTILS_HH

#include <gsl/gsl_matrix.h>

typedef double (*funMat)(gsl_matrix*, double inp);


double randfrom(double min, double max);

void setRandomMat(gsl_matrix* mat, double min, double max);

void printMatrix(const gsl_matrix* mat);

void multiplyMatrix(const gsl_matrix* a,const gsl_matrix* b, gsl_matrix* ret);

void fromArrayToColumn(gsl_matrix*a, const double vec[]);

void actSigmoid(gsl_matrix* inp);

void applyFunMatrix(gsl_matrix* inp, funMat fun);
#endif
