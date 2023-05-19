#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

struct connected_layer :public layer{
  void backward(network net);
  void forward(network net);
  void update(update_args);
  void statistics();
  void denormalize();

  
  layer* batch_normalize_layer =nullptr;
};

connected_layer* make_connected_layer(int batch, int inputs,
                                      int outputs, ACTIVATION activation,
                                      int batch_normalize, int adam);

#endif

