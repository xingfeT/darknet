#pragma once


#include "image.h"
#include "layer.h"
#include "network.h"

struct normalization_layer:public layer{
  void forward(network net);
  void backward(network net);
  void resize(int h, int w);
  void visualize(char *window) const;
};

normalization_layer* make_normalization_layer(int batch,
                                              int w, int h, int c, int size,
                                              float alpha, float beta, float kappa);
