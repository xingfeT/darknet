#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"



struct reorg_layer :public layer{
  void forward(network net);
  void backward(network net);
  void resize(int w, int h);
  
};
reorg_layer* make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
void backward_reorg_layer_gpu(layer l, network net);
#endif

#endif

