#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

l2norm_layer* make_l2norm_layer(int batch, int inputs){
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    l2norm_layer* l = new l2norm_layer();
    l->type = L2NORM;

    l->batch = batch;
    l->inputs = inputs;
    l->outputs = inputs;

    l->output = (float*)calloc(inputs*batch, sizeof(float));
    l->scales = (float*)calloc(inputs*batch, sizeof(float));
    l->delta = (float*)calloc(inputs*batch, sizeof(float));

    return l;
}

void l2norm_layer::forward( network net){
  l2norm_layer* l = this;
  copy_cpu(l->outputs*l->batch, net.input, 1, l->output, 1);
  l2normalize_cpu(l->output, l->scales, l->batch, l->out_c, l->out_w*l->out_h);
}

void l2norm_layer::backward( network net){
  l2norm_layer* l = this;
  axpy_cpu(l->inputs*l->batch, 1, l->delta, 1, net.delta, 1);
}


