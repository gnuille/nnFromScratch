#include "utils.h"

double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void setRandomMat(gsl_matrix* mat, double min, double max)
{
  int i, j;
  for(i = 0;i <mat->size1;i++){
    for(j = 0; j<mat->size2;j++){
      gsl_matrix_set(mat,i, j,randfrom(-1,1));
    }
  }
}

void printMatrix(gsl_matrix* mat){
  int i,j;
  for(i = 0; i<mat->size1; i++){
    for(j=0; j<mat->size2; j++){
      if(j) printf(" ");
      printf("%f", gsl_matrix_get(mat, i, j));
    }
    printf("\n");
  }
}
