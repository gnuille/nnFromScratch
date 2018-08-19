#include "neuralNetwork.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void loadNn(struct nn* neural,int n_layers, int sizes[], int input, int output){
  neural->n_layers = n_layers;
  neural->layer_sizes = sizes;
  neural->weights = malloc(sizeof(gsl_matrix*)*n_layers +1);
  neural->biases  = malloc(sizeof(gsl_matrix*)*n_layers +1);

  neural->weights[0] = gsl_matrix_calloc(neural->layer_sizes[0], input);
  neural->biases[0]  = gsl_matrix_calloc(neural->layer_sizes[0], 1    );
  int i;
  for (i = 1; i< n_layers; i++){
    neural->weights[i] = gsl_matrix_calloc(neural->layer_sizes[i-1], neural->layer_sizes[i]);
    neural->biases[i]  = gsl_matrix_calloc(neural->layer_sizes[i],1);
  }

  neural->weights[n_layers] = gsl_matrix_calloc(output,neural->layer_sizes[n_layers - 1]);
  neural->biases[n_layers]  = gsl_matrix_calloc(output,1);


  for(i = 0; i <= n_layers; ++i){
    setRandomMat(neural->weights[i], -1, 1);
    setRandomMat(neural->biases[i] , -1, 1);
  }

  printMatrix(neural->weights[0]);

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
  multiplyMatrix(inp, neural->weights[0]);
  gsl_matrix_add(inp, neural->biases[0]);
  int i;
  for(i = 1; i <=neural->n_layers; i++){
    multiplyMatrix(inp, neural->weights[i]);
    gsl_matrix_add(inp, neural->biases[i] );
  }
  printMatrix(inp);

}
