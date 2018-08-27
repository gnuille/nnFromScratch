#include "utils.h"

double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int randint(int n){
  if ((n - 1) == RAND_MAX) {
    return rand();
  }else{
    assert (n <= RAND_MAX);

    int end = RAND_MAX / n;
    assert (end > 0);
    end *= n;

    int r;

    while((r = rand()) >= end);

    return r % n;
  }
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
  if(a->size2 != b->size1){
    printf("error cannot multiply matrix with size: (%zu,%zu)*(%zu,%zu)",a->size1,a->size2,b->size1,b->size2);
  }
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

void fromArrayToLine(gsl_matrix*a, const double vec[]){
  int i;
  for(i = 0;i<a->size2;i++){
    gsl_matrix_set(a, 0, i, vec[i]);
  }
}


void applyFunMatrix(gsl_matrix* inp, funMat fun){
  int i,j;
  for(i = 0; i<inp->size1; i++){
    for(j=0; j<inp->size2; j++){
      gsl_matrix_set(inp, i, j, fun(gsl_matrix_get(inp,i,j)));
    }
  }
}

double sigmoid(double inp){
  return (1)/(1+exp(-inp));
}

double sigmoidDerivate(double inp){
  return inp*(1-inp);
}

double relu(double d){
  if(d > 0) return d;
  return 0;
}

double reluDerivative(double d){
  if(d > 0) return 1;
  return 0;
}

void printMatrixArray(gsl_matrix** mats, int size){
  int i;
  printf("Printing a matrix array...\n");
  for(i = 0; i<size; i++){
    printf("Matrix number %d\n",i);
    printMatrix(mats[i]);
  }
}

double* getRowAsArray(gsl_matrix* mat, int row){
  double* ret = malloc(sizeof(double)*mat->size2);
  int i;
  for (i = 0; i<mat->size2; i++){
    ret[i] = gsl_matrix_get(mat, row, i);
  }
  return ret;
}
