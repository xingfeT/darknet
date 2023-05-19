#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

struct logistic_layer :public layer{
  void forward( network net);
  void backward(network net);

};

logistic_layer* make_logistic_layer(int batch, int inputs);

#endif
