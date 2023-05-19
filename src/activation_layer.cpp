#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




activation_layer* make_activation_layer(int batch, int inputs, ACTIVATION activation){
  activation_layer* l = new activation_layer();
  l->type = ACTIVE;

  l->inputs = inputs;
  l->outputs = inputs;
  l->batch=batch;


  l->output = (float*)calloc(batch*inputs, sizeof(float*));
  l->delta = (float*)calloc(batch*inputs, sizeof(float*));

  l->activation = activation;
  fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
  return l;
}
