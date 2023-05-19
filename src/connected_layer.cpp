#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer*
make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam){
    int i;
    connected_layer* l = new connected_layer();
    l->learning_rate_scale = 1;
    l->type = CONNECTED;

    l->inputs = inputs;
    l->outputs = outputs;
    l->batch=batch;
    l->batch_normalize = batch_normalize;
    l->h = 1;
    l->w = 1;
    l->c = inputs;
    l->out_h = 1;
    l->out_w = 1;
    l->out_c = outputs;

    l->output = (float*)calloc(batch*outputs, sizeof(float));
    l->delta = (float*)calloc(batch*outputs, sizeof(float));

    l->weight_updates = (float*)calloc(inputs*outputs, sizeof(float));
    l->bias_updates = (float*)calloc(outputs, sizeof(float));

    l->weights = (float*)calloc(outputs*inputs, sizeof(float));
    l->biases = (float*)calloc(outputs, sizeof(float));

    float scale = sqrt(2.0/inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l->weights[i] = scale*rand_uniform(-1, 1);
    }

    for(i = 0; i < outputs; ++i){
        l->biases[i] = 0;
    }

    if(adam) init_adam(l);

    
    if(batch_normalize) init_batch_normalize(l, outputs, batch);

    l->activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void connected_layer::update(update_args a){
  connected_layer* l = this;
  
    float learning_rate = a.learning_rate*l->learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l->outputs, learning_rate/batch, l->bias_updates, 1, l->biases, 1);
    scal_cpu(l->outputs, momentum, l->bias_updates, 1);

    if(l->batch_normalize){
        axpy_cpu(l->outputs, learning_rate/batch, l->scale_updates, 1, l->scales, 1);
        scal_cpu(l->outputs, momentum, l->scale_updates, 1);
    }

    axpy_cpu(l->inputs*l->outputs, -decay*batch, l->weights, 1, l->weight_updates, 1);
    axpy_cpu(l->inputs*l->outputs, learning_rate/batch, l->weight_updates, 1, l->weights, 1);
    scal_cpu(l->inputs*l->outputs, momentum, l->weight_updates, 1);
}

void connected_layer::forward(network net){
  connected_layer* l = this;
  fill_cpu(l->outputs*l->batch, 0, l->output, 1);
  int m = l->batch;
  int k = l->inputs;
  int n = l->outputs;
  float *a = net.input;
  float *b = l->weights;
  float *c = l->output;
  gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
  if(l->batch_normalize){
    l->batch_normalize_layer->forward(net);
  } else {
    add_bias(l->output, l->biases, l->batch, l->outputs, 1);
  }
  activate_array(l->output, l->outputs*l->batch, l->activation);
}

void connected_layer::backward(network net){
  connected_layer* l = this;
  
    gradient_array(l->output, l->outputs*l->batch, l->activation, l->delta);

    if(l->batch_normalize){
      l->batch_normalize_layer->backward(net);
    } else {
        backward_bias(l->bias_updates, l->delta, l->batch, l->outputs, 1);
    }

    int m = l->outputs;
    int k = l->batch;
    int n = l->inputs;
    float *a = l->delta;
    float *b = net.input;
    float *c = l->weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l->batch;
    k = l->outputs;
    n = l->inputs;

    a = l->delta;
    b = l->weights;
    c = net.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}


void connected_layer::denormalize(){
  connected_layer* l = this;
    int i, j;
    for(i = 0; i < l->outputs; ++i){
      float scale = l->scales[i]/std::sqrt(l->rolling_variance[i] + .000001);
        for(j = 0; j < l->inputs; ++j){
            l->weights[i*l->inputs + j] *= scale;
        }
        l->biases[i] -= l->rolling_mean[i] * scale;
        l->scales[i] = 1;
        l->rolling_mean[i] = 0;
        l->rolling_variance[i] = 1;
    }
}


void connected_layer::statistics(){
  connected_layer* l = this;
  
    if(l->batch_normalize){
        printf("Scales ");
        print_statistics(l->scales, l->outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l->rolling_mean, l->outputs);
           printf("Rolling Variance ");
           print_statistics(l->rolling_variance, l->outputs);
         */
    }
    printf("Biases ");
    print_statistics(l->biases, l->outputs);
    printf("Weights ");
    print_statistics(l->weights, l->outputs);
}

