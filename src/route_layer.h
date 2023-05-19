#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

struct route_layer :public layer{
  void forward( network net) =0;
  void backward( network net) =0;
  void resize(network *net);
};

route_layer* make_route_layer(int batch, int n, int *input_layers, int *input_size);

#endif
