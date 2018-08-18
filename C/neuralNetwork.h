#ifndef NN_HEADER
#define NN_HEADER
#include <gsl/gsl_matrix.h>

struct nn {
  int n_layers;
  int* layer_sizes;
  gsl_matrix** weights;
  gsl_matrix** biases;
};

void loadNn(struct nn* neural,int n_layers, int* sizes, int input, int output);

void printNn(struct nn* neural, int debugLvl);

#endif
