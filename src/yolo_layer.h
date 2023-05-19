#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"


struct yolo_layer:public layer{
  void forward(network net);
  void backward(network net);
  void resize(int w, int h);
  int yolo_num_detections(float thresh) ;
  void avg_flipped_yolo();
};

yolo_layer* make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);

int get_yolo_detections(yolo_layer* l, int w, int h, int netw, int neth, float thresh,
                        int *map, int relative, detection *dets);


#endif
