#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer* make_softmax_layer(int batch, int inputs, int groups){
  assert(inputs%groups == 0);
  fprintf(stderr, "softmax                                        %4d\n",  inputs);
  softmax_layer* l = new softmax_layer();
  l->type = SOFTMAX;

  l->batch = batch;
  l->inputs = inputs;
  l->outputs = inputs;

  l->groups = groups;

  l->loss = (float*)calloc(inputs*batch, sizeof(float));
  l->output = (float*)calloc(inputs*batch, sizeof(float));

  l->delta = (float*)calloc(inputs*batch, sizeof(float));
  
  l->cost = (float*)calloc(1, sizeof(float));

  return l;
}

void softmax_layer::forward(network net){
  softmax_layer* l = this;
  
  if(l->softmax_tree){
    int i;
    int count = 0;
    for (i = 0; i < l->softmax_tree->groups; ++i) {
      int group_size = l->softmax_tree->group_size[i];
      softmax_cpu(net.input + count, group_size, l->batch, l->inputs, 1, 0, 1, l->temperature, l->output + count);
      count += group_size;
    }
  } else {
    softmax_cpu(net.input, l->inputs/l->groups, l->batch, l->inputs, l->groups, l->inputs/l->groups, 1, l->temperature, l->output);
  }
  
    if(net.truth && !l->noloss){
      softmax_x_ent_cpu(l->batch*l->inputs, l->output, net.truth, l->delta, l->loss);
      l->cost[0] = sum_array(l->loss, l->batch*l->inputs);
    }
}

void softmax_layer::backward(network net){
  softmax_layer* l = this;
  axpy_cpu(l->inputs*l->batch, 1, l->delta, 1, net.delta, 1);
}
