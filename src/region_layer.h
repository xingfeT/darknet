#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"


struct region_layer :public layer{
  void forward(network net) ;
  void backward(network net);
  void resize(int w, int h);
};

region_layer* make_region_layer(int batch, int w, int h, int n, int classes, int coords);


#endif
