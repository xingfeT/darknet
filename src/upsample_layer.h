#pragma once

#include "darknet.h"


struct upsample_layer :public layer{
  void forward(  network net) = 0;
  void backward( network net) = 0;
  void resize(int w, int h);
};

upsample_layer* make_upsample_layer(int batch, int w, int h, int c, int stride);


#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net);
void backward_upsample_layer_gpu(const layer l, network net);
#endif
