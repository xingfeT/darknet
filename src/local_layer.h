#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"



#ifdef GPU
void forward_local_layer_gpu(local_layer layer, network net);
void backward_local_layer_gpu(local_layer layer, network net);
void update_local_layer_gpu(local_layer layer, update_args a);

void push_local_layer(local_layer layer);
void pull_local_layer(local_layer layer);
#endif


struct local_layer :public layer{
  void forward(network net);
  void backward(network net);
  void update(update_args a);
 
};
local_layer* make_local_layer(int batch, int h, int w, int c, int n,
                              int size, int stride, int pad, ACTIVATION activation);



void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#endif

