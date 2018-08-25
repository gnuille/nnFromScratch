#include "neuralNetwork.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void loadNn(struct nn* neural,int n_layers, int sizes[], int input, int output, actFunc act){
  neural->n_layers = n_layers;
  neural->layer_sizes = sizes;
  neural->weights = malloc(sizeof(gsl_matrix*)*(n_layers +1));
  neural->biases  = malloc(sizeof(gsl_matrix*)*(n_layers +1));
  neural->results = malloc(sizeof(gsl_matrix*)*(n_layers +1));

  neural->weights[0] = gsl_matrix_calloc(neural->layer_sizes[0], input);
  neural->biases[0]  = gsl_matrix_calloc(neural->layer_sizes[0], 1    );
  int i;
  for (i = 1; i< n_layers; i++){
    neural->weights[i] = gsl_matrix_calloc(neural->layer_sizes[i],neural->layer_sizes[i-1]);
    neural->biases[i]  = gsl_matrix_calloc(neural->layer_sizes[i],1);
  }

  neural->weights[n_layers] = gsl_matrix_calloc(output,neural->layer_sizes[n_layers-1]);
  neural->biases[n_layers]  = gsl_matrix_calloc(output,1);


  for(i = 0; i <= n_layers; ++i){
    setRandomMat(neural->weights[i], -1, 1);
    setRandomMat(neural->biases[i] , -1, 1);
  }

  if(act == NULL){
    neural->act = sigmoid;
  }

  for(i = 0; i < n_layers; ++i){
    neural->results[i] = gsl_matrix_calloc(neural->layer_sizes[i], 1);
  }
  neural->results[n_layers] = gsl_matrix_calloc(output, 1);

}

void printNn(struct nn* neural, int debugLvl){
  printf("Printing neural network information \n");
  printf("Number of internal layers: %i\n", neural->n_layers);
  printf("Internal layer sizes:");
  int i;
  for(i = 0; i<neural->n_layers;++i)
  {
    if(i) printf(",");
    printf("%d",neural->layer_sizes[i]);
  }
  printf("\n");
  if (!debugLvl) return;
  printf("Weights:\n");
  for(i = 0; i<=neural->n_layers; ++i)
  {
    printf("Weight from layer number %d\n", i);
    printMatrix(neural->weights[i]);
  }
  printf("Biases:\n");
  for(i = 0; i<=neural->n_layers; ++i)
  {
    printf("Bias from layer number %d\n", i);
    printMatrix(neural->biases[i]);
  }
  printf("\n");
}

void predictNn(struct nn* neural, double input[]){
  gsl_matrix *inp = gsl_matrix_alloc(neural->weights[0]->size2, 1); //conversion from input vector to matrix
  fromArrayToColumn(inp, input);
  multiplyMatrix(neural->weights[0], inp, inp);
  gsl_matrix_add(inp, neural->biases[0]);
  int i;
  for(i = 1; i <=neural->n_layers; i++){
    multiplyMatrix(neural->weights[i], inp, inp);
    gsl_matrix_add(inp, neural->biases[i]);
  }
  applyFunMatrix(inp, neural->act);

  printf("Result:\n");
  printMatrix(inp);
  gsl_matrix_free(inp);
}

void stepTrain(struct nn* neural, double* input, double* output){
  //transform input and output vectors to matrices for easly working
  gsl_matrix *inp = gsl_matrix_alloc(neural->weights[0]->size2, 1);
  gsl_matrix *out = gsl_matrix_alloc(neural->weights[neural->n_layers]->size1, 1);

  fromArrayToColumn(inp, input);
  fromArrayToColumn(out, output);

  //fastforward like prediction and store temporary results for calculating errors
  int i;
  for(i = 0; i <=neural->n_layers; i++){
    multiplyMatrix(neural->weights[i], inp, inp);
    gsl_matrix_add(inp, neural->biases[i]);
    if(i == neural->n_layers) applyFunMatrix(inp, neural->act);
    gsl_matrix_memcpy(neural->results[i],inp);
  }

  //array of matrices for errors of each layer
  gsl_matrix** errors = malloc(sizeof(gsl_matrix*)*(neural->n_layers+1));
  for(i = 0; i<=neural->n_layers; i++){
    errors[i] = gsl_matrix_alloc(neural->biases[i]->size1,1);
  }
  //calculate the errors
  gsl_matrix_memcpy(errors[neural->n_layers],out);
  gsl_matrix_sub(errors[neural->n_layers], inp);
  for (i = neural->n_layers; i > 0; i--){
    gsl_matrix* weightT = gsl_matrix_alloc(neural->weights[i]->size2,neural->weights[i]->size1);
    gsl_matrix_transpose_memcpy(weightT,neural->weights[i]);
    multiplyMatrix(weightT, errors[i], errors[i-1]);
    gsl_matrix_free(weightT);
  }
  printMatrixArray(errors, neural->n_layers+1);
}
