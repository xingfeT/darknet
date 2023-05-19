#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET


struct lstm_layer:public layer{
  void forward(network net) ;
  void backward(network net);
  void update(update_args a);
  void increment_layer(int steps);
};

lstm_layer* make_lstm_layer(int batch,
                            int inputs, int outputs, int steps, int batch_normalize, int adam);


#endif
