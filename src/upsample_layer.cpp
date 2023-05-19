#include "upsample_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

upsample_layer* make_upsample_layer(int batch, int w, int h, int c, int stride){
  upsample_layer*  l = new upsample_layer();
  
    l->type = UPSAMPLE;
    l->batch = batch;
    l->w = w;
    l->h = h;
    l->c = c;
    l->out_w = w*stride;
    l->out_h = h*stride;
    l->out_c = c;
    if(stride < 0){
        stride = -stride;
        l->reverse=1;
        l->out_w = w/stride;
        l->out_h = h/stride;
    }
    l->stride = stride;
    l->outputs = l->out_w*l->out_h*l->out_c;

    l->inputs = l->w*l->h*l->c;
    l->delta =  (float*)calloc(l->outputs*batch, sizeof(float));
    l->output = (float*)calloc(l->outputs*batch, sizeof(float));;

    
    if(l->reverse) fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l->out_w, l->out_h, l->out_c);
    else fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l->out_w, l->out_h, l->out_c);
    return l;
}

void upsample_layer::resize(int w, int h){
  upsample_layer* l = this;
  
  l->w = w;
  l->h = h;
  l->out_w = w*l->stride;
  l->out_h = h*l->stride;
  if(l->reverse){
    l->out_w = w/l->stride;
    l->out_h = h/l->stride;
  }
  l->outputs = l->out_w*l->out_h*l->out_c;
  l->inputs = l->h*l->w*l->c;
  l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
  l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));

}

void upsample_layer::forward(network net){
  upsample_layer* l = this;
  
    fill_cpu(l->outputs*l->batch, 0, l->output, 1);
    if(l->reverse){
        upsample_cpu(l->output, l->out_w, l->out_h, l->c, l->batch, l->stride, 0, l->scale, net.input);
    }else{
        upsample_cpu(net.input, l->w, l->h, l->c, l->batch, l->stride, 1, l->scale, l->output);
    }
}

void upsample_layer::backward(network net){
  upsample_layer* l = this;
  
  if(l->reverse){
    upsample_cpu(l->delta, l->out_w, l->out_h, l->c, l->batch, l->stride, 1, l->scale, net.delta);
  }else{
    upsample_cpu(net.delta, l->w, l->h, l->c, l->batch, l->stride, 0, l->scale, l->delta);
  }
}

