#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"


struct batchnorm_layer :public layer{
   void forward(network net);
   void backward(network net);
   void resize( int w, int h);
};

batchnorm_layer* make_batchnorm_layer(int batch, int w, int h, int c);


#endif
