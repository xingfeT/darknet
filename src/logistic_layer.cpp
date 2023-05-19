#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

logistic_layer* make_logistic_layer(int batch, int inputs){
  fprintf(stderr, "logistic x entropy                             %4d\n",  inputs);
  logistic_layer* l = new logistic_layer();
  
  l->type = LOGXENT;
  l->batch = batch;
  l->inputs = inputs;
  l->outputs = inputs;
  
  l->loss = (float*)calloc(inputs*batch, sizeof(float));
  l->output = (float*)calloc(inputs*batch, sizeof(float));
  l->delta = (float*)calloc(inputs*batch, sizeof(float));
  
  l->cost = (float*)calloc(1, sizeof(float));

  return l;
}

void logistic_layer::forward( network net){
  logistic_layer* l = this;
  
  copy_cpu(l->outputs*l->batch, net.input, 1, l->output, 1);
  activate_array(l->output, l->outputs*l->batch, LOGISTIC);
  if(net.truth){
    logistic_x_ent_cpu(l->batch*l->inputs, l->output, net.truth, l->delta, l->loss);
    l->cost[0] = sum_array(l->loss, l->batch*l->inputs);
  }
}

void logistic_layer::backward(network net){
  logistic_layer* l = this;
  axpy_cpu(l->inputs*l->batch, 1, l->delta, 1, net.delta, 1);
}

