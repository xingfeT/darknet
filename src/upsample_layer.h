#pragma once

#include "darknet.h"


struct upsample_layer :public layer{
  void forward(  network net);
  void backward( network net);

  void resize(int w, int h);
};

upsample_layer* make_upsample_layer(int batch, int w, int h, int c,
                                    int stride);
