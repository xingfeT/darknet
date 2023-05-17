#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include "layer.h"
#include "network.h"


struct detection_layer :public layer{
  void forward( network net);
  void backward( network net);
};

detection_layer* make_detection_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);


#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network net);
void backward_detection_layer_gpu(detection_layer l, network net);
#endif

#endif
