#ifndef NN_HEADER
#define NN_HEADER
#include <gsl/gsl_matrix.h>

typedef void (*actFunc)(gsl_matrix*);

struct nn {
  int n_layers;
  int* layer_sizes;
  gsl_matrix** weights;
  gsl_matrix** biases;
  actFunc act;
};

void loadNn(struct nn* neural,int n_layers, int* sizes, int input, int output, actFunc act);

void printNn(struct nn* neural, int debugLvl);

void predictNn(struct nn* neural, double* input);

#endif
