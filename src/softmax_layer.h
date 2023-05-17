#pragma once
#include "layer.h"
#include "network.h"

void softmax_array(float *input, int n, float temp, float *output);


struct softmax_layer : public layer{
  void forward( network net) const ;
  void backward( network net) const ;
};

softmax_layer* make_softmax_layer(int batch, int inputs, int groups);


#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
#endif
