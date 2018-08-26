#ifndef NN_HEADER
#define NN_HEADER
#include <gsl/gsl_matrix.h>

typedef double (*actFunc)(double);

struct nn {
  int n_layers;
  int* layer_sizes;
  gsl_matrix** weights;
  gsl_matrix** biases;
  gsl_matrix** results;
  actFunc act;
  actFunc derivate;
};

void loadNn(struct nn* neural,int n_layers, int* sizes, int input, int output, actFunc act, actFunc derivate);

void printNn(struct nn* neural, int debugLvl);

void predictNn(struct nn* neural, double* input);

void trainNn(struct nn* neural, gsl_matrix* inputs, gsl_matrix* outputs, int size, double learning_rate, int batches);

void stepTrain(struct nn* neural, double* input, double* output, double learning_rate);

#endif
