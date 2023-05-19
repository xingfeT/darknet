#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

struct deconvolutional_layer :public layer{
  void forward(struct network);
  void backward(struct network);
  
  void update(update_args);
  void bilinear_init();
  void resize(int, int);
  void denormalize();
  
  size_t workspaceSize();
};

deconvolutional_layer* make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size,
                                                  int stride, int padding,
                                                  ACTIVATION activation, int batch_normalize, int adam);


/* void resize_deconvolutional_layer(layer *l, int h, int w); */
/* void forward_deconvolutional_layer(const layer l, network net); */
/* void update_deconvolutional_layer(layer l, update_args a); */
/* void backward_deconvolutional_layer(layer l, network net); */

#endif

