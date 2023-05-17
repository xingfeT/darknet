#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

//typedef layer convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

struct ConvolutionalLayer :public layer{
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
};

typedef ConvolutionalLayer convolutional_layer;


ConvolutionalLayer* make_convolutional_layer(int batch, int h, int w, int c, int n,
                                              int groups, int size, int stride,
                                              int padding, ACTIVATION activation,
                                              int batch_normalize, int binary, int xnor, int adam);

void binarize_weights(float *weights, int n, int size, float *binary);
void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);
void binarize_weights2(float *weights, int n, int size, char *binary, float *scales);


#endif

