#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"



struct avgpool_layer :public layer{
  image get_image();
  void resize(int w, int h);
  void forward(struct network) ;
  void backward(struct network);
};

avgpool_layer* make_avgpool_layer(int batch, int w, int h, int c);

#endif

