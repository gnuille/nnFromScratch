#include "neuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(){
  srand(time(NULL));

  struct nn neural;
  int sizes[] = {3, 4};
  loadNn(&neural, 2, sizes ,2, 1, NULL, NULL);
  double inp[] = {0.5, 0.4};
  double out[] = {0.1};
  int i;
  //obviusly overfitted, not measuring NN performance just testing if it works.
  for(i = 0; i<10000;i++){
    stepTrain(&neural, inp, out, 0.1);
  }
  predictNn(&neural, inp);
}
