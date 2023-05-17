
#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
struct gru_layer :public layer{
  void forward( network state);
  void backward( network state);
  void update( update_args a);
};

gru_layer* make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);



#ifdef GPU
void forward_gru_layer_gpu(layer l, network state);
void backward_gru_layer_gpu(layer l, network state);
void update_gru_layer_gpu(layer l, update_args a);
void push_gru_layer(layer l);
void pull_gru_layer(layer l);
#endif

#endif

