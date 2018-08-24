#include "neuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(){
  srand(time(NULL));

  struct nn neural;
  int sizes[] = {3, 4};
  loadNn(&neural, 2, sizes ,2, 1, NULL);
  double inp[] = {0.5, 0.4};
  double out[] = {0.1};
  stepTrain(&neural, inp, out);
}
