#pragma once
#include "layer.h"
#include "network.h"

void softmax_array(float *input, int n, float temp, float *output);


struct softmax_layer : public layer{
  void forward( network net) ;
  void backward( network net)  ;
  void update(update_args){}

};

softmax_layer* make_softmax_layer(int batch, int inputs, int groups);
