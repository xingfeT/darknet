#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"


struct maxpool_layer:public layer{
  void forward(network net) const ;
  void backward(network net) const ;
  void resize(maxpool_layer *l, int w, int h);
  image get_image() const;
};


maxpool_layer* make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);



#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);
#endif

#endif

