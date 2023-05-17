#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"



struct AvgPoolLayer :public layer{
  image get_image();
  void resize_layer(int w, int h);
};

AvgPoolLayer* make_avgpool_layer(int batch, int w, int h, int c);
typedef AvgPoolLayer avgpool_layer;




/* void forward_avgpool_layer(const avgpool_layer l, network net); */
/* void backward_avgpool_layer(const avgpool_layer l, network net); */

/* #ifdef GPU */
/* void forward_avgpool_layer_gpu(avgpool_layer l, network net); */
/* void backward_avgpool_layer_gpu(avgpool_layer l, network net); */
/* #endif */

#endif

