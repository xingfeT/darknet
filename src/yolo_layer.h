#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"


struct yolo_layer:public layer{
  void forward(network net);
  void backward(network net);
  void resize(int w, int h);
  int yolo_num_detections(float thresh) const ;
};
yolo_layer* make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);




#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network net);
void backward_yolo_layer_gpu(layer l, network net);
#endif

#endif
