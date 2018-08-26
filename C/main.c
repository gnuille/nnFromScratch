#include "neuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_matrix.h>


int main(){
  srand(time(NULL));

  struct nn neural;
  int sizes[] = {2};
  loadNn(&neural, 1, sizes ,2, 1, NULL, NULL);

  gsl_matrix* inputs = gsl_matrix_alloc(4, 2);
  gsl_matrix* outputs = gsl_matrix_alloc(4,1);

  gsl_matrix_set(inputs, 0, 0, 0);
  gsl_matrix_set(inputs, 0, 1, 0);
  gsl_matrix_set(inputs, 1, 0, 0);
  gsl_matrix_set(inputs, 1, 1, 1);
  gsl_matrix_set(inputs, 2, 0, 1);
  gsl_matrix_set(inputs, 2, 1, 0);
  gsl_matrix_set(inputs, 3, 0, 1);
  gsl_matrix_set(inputs, 3, 1, 1);

  gsl_matrix_set(outputs, 0, 0, 0);
  gsl_matrix_set(outputs, 1, 0, 0);
  gsl_matrix_set(outputs, 2, 0, 0);
  gsl_matrix_set(outputs, 3, 0, 1);

  trainNn(&neural, inputs, outputs, 4, 0.0000000001, 1000000);

  double test[] = {0, 1};
  predictNn(&neural, test);


}
