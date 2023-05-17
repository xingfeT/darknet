#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET


struct lstm_layer:public layer{
  void forward(network net) const ; 
  void update(update_args a);
};

lstm_layer* make_lstm_layer(int batch,
                            int inputs, int outputs, int steps, int batch_normalize, int adam);


#ifdef GPU
void forward_lstm_layer_gpu(layer l, network net);
void backward_lstm_layer_gpu(layer l, network net);
void update_lstm_layer_gpu(layer l, update_args a); 

#endif
#endif
