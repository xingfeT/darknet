#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s){
  if (strcmp(s, "seg")==0) return SEG;
  if (strcmp(s, "sse")==0) return SSE;
  if (strcmp(s, "masked")==0) return MASKED;
  if (strcmp(s, "smooth")==0) return SMOOTH;
  if (strcmp(s, "L1")==0) return L1;
  if (strcmp(s, "wgan")==0) return WGAN;
  fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
  return SSE;
}

const char *get_cost_string(COST_TYPE a){
  switch(a){
  case SEG:
    return "seg";
  case SSE:
    return "sse";
  case MASKED:
    return "masked";
  case SMOOTH:
    return "smooth";
  case L1:
    return "L1";
  case WGAN:
    return "wgan";
  }
  return "sse";
}

cost_layer* make_cost_layer(int batch, int inputs,
                            COST_TYPE cost_type,
                            float scale){
  fprintf(stderr, "cost                                           %4d\n",  inputs);
  cost_layer* l =  new cost_layer();
  l->type = COST;

  l->scale = scale;
  l->batch = batch;
  
  l->cost_type = cost_type;

  l->cost = (float*)calloc(1, sizeof(float));
  l->resize(inputs);

  return l;
}

void cost_layer::resize(int inputs){
  cost_layer* l = this;
  l->inputs = l->outputs = inputs;
  l->delta = (float*)realloc(l->delta, inputs*l->batch*sizeof(float));
  l->output = (float*)realloc(l->output, inputs*l->batch*sizeof(float));
}

void cost_layer::forward(network net){
  cost_layer* l = this;
  
  if (!net.truth) return;
  if(l->cost_type == MASKED){
    int i;
    for(i = 0; i < l->batch*l->inputs; ++i){
      if(net.truth[i] == SECRET_NUM) net.input[i] = SECRET_NUM;
    }
  }
  if(l->cost_type == SMOOTH){
    smooth_l1_cpu(l->batch*l->inputs, net.input, net.truth, l->delta, l->output);
  }else if(l->cost_type == L1){
    l1_cpu(l->batch*l->inputs, net.input, net.truth, l->delta, l->output);
  } else {
    l2_cpu(l->batch*l->inputs, net.input, net.truth, l->delta, l->output);
  }
  l->cost[0] = sum_array(l->output, l->batch*l->inputs);
}

void cost_layer::backward(network net){
  cost_layer* l = this;
  axpy_cpu(l->batch*l->inputs, l->scale, l->delta, 1, net.delta, 1);
}


