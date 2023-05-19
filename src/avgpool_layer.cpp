#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

avgpool_layer* make_avgpool_layer(int batch, int w, int h, int c){
  fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    
  avgpool_layer* l = new avgpool_layer();
    
    l->type = AVGPOOL;
    l->batch = batch;
    l->h = h;
    l->w = w;
    l->c = c;
    l->out_w = 1;
    l->out_h = 1;
    l->out_c = c;
    l->outputs = l->out_c;
    l->inputs = h*w*c;
    int output_size = l->outputs * batch;

    l->output =  (float*)calloc(output_size, sizeof(float));
    l->delta =   (float*)calloc(output_size, sizeof(float));
    return l;
}

void avgpool_layer::resize(int w, int h){
  avgpool_layer* l = this;
  
  l->w = w;
  l->h = h;
  l->inputs = h*w*l->c;
}

void avgpool_layer::forward(network net){
  avgpool_layer* l = this;
    int b,i,k;

    for(b = 0; b < l->batch; ++b){
        for(k = 0; k < l->c; ++k){
            int out_index = k + b*l->c;
            l->output[out_index] = 0;
            for(i = 0; i < l->h*l->w; ++i){
                int in_index = i + l->h*l->w*(k + b*l->c);
                l->output[out_index] += net.input[in_index];
            }
            l->output[out_index] /= l->h*l->w;
        }
    }
}

void avgpool_layer::backward(network net){
  avgpool_layer* l = this;
  
  int b,i,k;
  
  for(b = 0; b < l->batch; ++b){
    for(k = 0; k < l->c; ++k){
      int out_index = k + b*l->c;
      for(i = 0; i < l->h*l->w; ++i){
        int in_index = i + l->h*l->w*(k + b*l->c);
        net.delta[in_index] += l->delta[out_index] / (l->h*l->w);
      }
    }
  }
}

