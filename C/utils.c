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

void printMatrix(const gsl_matrix* mat){
  int i,j;
  for(i = 0; i<mat->size1; i++){
    for(j=0; j<mat->size2; j++){
      if(j) printf(" ");
      printf("%f", gsl_matrix_get(mat, i, j));
    }
    printf("\n");
  }
}

void multiplyMatrix(const gsl_matrix* a1,const gsl_matrix* b1,gsl_matrix *ret){
  gsl_matrix* a = gsl_matrix_alloc(a1->size1,a1->size2);
  gsl_matrix* b = gsl_matrix_alloc(b1->size1,b1->size2);
  gsl_matrix_memcpy(a, a1);
  gsl_matrix_memcpy(b, b1);
  gsl_matrix_free(ret);
  ret = gsl_matrix_calloc(a->size1,b->size2);
  int n = a->size2;
  int m = a->size1;
  int k = b->size2;
  int i,j,it;
  for(i = 0; i<m; i++){
    for(j = 0; j<k; j++){
      double sum = 0;
      for(it = 0; it<n; ++it){
        sum += gsl_matrix_get(a, i, it)*gsl_matrix_get(b, it, j);
      }
      gsl_matrix_set(ret,i,j,sum);
    }
  }
  gsl_matrix_free(a);
  gsl_matrix_free(b);
}

void fromArrayToColumn(gsl_matrix*a, const double vec[]){
  int i;
  for(i = 0;i<a->size1;i++){
    gsl_matrix_set(a, i, 0, vec[i]);
  }
}
