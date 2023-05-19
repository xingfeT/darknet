
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET


struct rnn_layer :public layer{
  void forward(network net);
  void backward(network net);
  void update(update_args a);
  void increment_layer(int steps);

};

rnn_layer* make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation,
                          int batch_normalize, int adam);


#endif
