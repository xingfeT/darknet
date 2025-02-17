#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"


struct convolutional_layer :public layer{
  void resize(int w, int h);

  void forward(network net);
  void update(update_args a);
  void backward(network net);

  image *visualize(char *window, image *prev_weights);
  void swap_binary();
  image get_image();
  image get_delta();
  image get_weight(int i);

  int out_height();
  int out_width();
  size_t workspaceSize();
  void denormalize();

  void rgbgr_weights();
  void rescale_weights(float scale, float trans);
  image *get_weights();

  layer* batch_normalize_layer;
  layer* batchnorm_layer;

};




convolutional_layer* make_convolutional_layer(int batch, int h, int w, int c, int n,
                                              int groups, int size, int stride,
                                              int padding, ACTIVATION activation,
                                              int batch_normalize, int binary, int xnor, int adam);

void binarize_weights(float *weights, int n, int size, float *binary);
void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);


#endif
