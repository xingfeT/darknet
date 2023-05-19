#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer* make_dropout_layer(int batch, int inputs, float probability){
  dropout_layer* l = new dropout_layer();
  l->type = DROPOUT;
  l->probability = probability;
  l->inputs = inputs;
  l->outputs = inputs;
  l->batch = batch;
  l->rand = (float*)calloc(inputs*batch, sizeof(float));
  l->scale = 1.0/(1.0-probability);
  fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
  return l;
}

void dropout_layer::resize(int inputs){
  dropout_layer* l = this;
  l->rand = (float*)realloc(l->rand, l->inputs*l->batch*sizeof(float));
}

void dropout_layer::forward(network net){
  dropout_layer* l = this;
  int i;
  if (!net.train) return;

  for(i = 0; i < l->batch * l->inputs; ++i){
    float r = rand_uniform(0, 1);
    l->rand[i] = r;
    if(r < l->probability) net.input[i] = 0;
    else net.input[i] *= l->scale;
  }
}

void dropout_layer::backward(network net){
  dropout_layer* l = this;
  if(!net.delta) return;
  for(int i = 0; i < l->batch * l->inputs; ++i){
    float r = l->rand[i];
    if(r < l->probability) net.delta[i] = 0;
    else net.delta[i] *= l->scale;
  }
}

