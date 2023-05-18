#ifndef ISEG_LAYER_H
#define ISEG_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"


struct iseg_layer :public layer{
  void forward(network net);
  void backward(network net);
  void resize(int w, int h);

};
iseg_layer* make_iseg_layer(int batch, int w, int h, int classes, int ids);


void resize_iseg_layer(layer *l, int w, int h);
int iseg_num_detections(layer *l, float thresh);

#ifdef GPU
void forward_iseg_layer_gpu(const layer l, network net);
void backward_iseg_layer_gpu(layer l, network net);
#endif

#endif
