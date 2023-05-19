#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"


struct maxpool_layer:public layer{
  void forward(network net) ;
  void backward(network net) ;
  void resize( int w, int h);
  
  image get_image();
  image get_delta();
};


maxpool_layer* make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);



#endif

