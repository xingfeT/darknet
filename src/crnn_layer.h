
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
struct crnn_layer  :public layer{

  void forward( network net);
  void backward( network net);
  void update(update_args a);

  void increment_layer(int steps);

};

crnn_layer* make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);


#endif
