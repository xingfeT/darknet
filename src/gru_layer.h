
#pragma once 
#include "activations.h"
#include "layer.h"
#include "network.h"

struct gru_layer :public layer{
  void forward( network state);
  void backward( network state);
  void update( update_args a);
  void increment_layer(int steps);
  
};

gru_layer* make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);


