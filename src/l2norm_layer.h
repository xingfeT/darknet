#ifndef L2NORM_LAYER_H
#define L2NORM_LAYER_H
#include "layer.h"
#include "network.h"

struct l2norm_layer :public layer{
  void backward( network net);
  void forward( network net);
};


l2norm_layer* make_l2norm_layer(int batch, int inputs);


#endif
