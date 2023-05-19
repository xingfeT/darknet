#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

struct dropout_layer :public layer {
  void forward(network net);
  void backward(network net);
  void resize(int inputs);
};

dropout_layer* make_dropout_layer(int batch, int inputs, float probability);


#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer l, network net);
void backward_dropout_layer_gpu(dropout_layer l, network net);

#endif
#endif
