#include "reorg_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>


reorg_layer*  make_reorg_layer(int batch, int w, int h, int c,
                       int stride, int reverse, int flatten, int extra){
  reorg_layer* l =  new reorg_layer();

    l->type = REORG;
    l->batch = batch;
    l->stride = stride;
    l->extra = extra;
    l->h = h;
    l->w = w;
    l->c = c;
    l->flatten = flatten;
    if(reverse){
      l->out_w = w*stride;
      l->out_h = h*stride;
      l->out_c = c/(stride*stride);
    }else{
      l->out_w = w/stride;
      l->out_h = h/stride;
      l->out_c = c*(stride*stride);
    }
    l->reverse = reverse;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = h*w*c;
    if(l->extra){
      l->out_w = l->out_h = l->out_c = 0;
      l->outputs = l->inputs + l->extra;
    }

    if(extra){
      fprintf(stderr, "reorg              %4d   ->  %4d\n",  l->inputs, l->outputs);
    } else {
      fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
              stride, w, h, c, l->out_w, l->out_h, l->out_c);
    }

    int output_size = l->outputs * batch;
    l->output =  (float*)calloc(output_size, sizeof(float));
    l->delta =   (float*)calloc(output_size, sizeof(float));

    return l;
}

void reorg_layer::resize(int w, int h){
  reorg_layer * l = this;

    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse){
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }else{
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = (float*)realloc(l->output, output_size * sizeof(float));
    l->delta = (float*)realloc(l->delta, output_size * sizeof(float));

}

void reorg_layer::forward(network net){
  reorg_layer* l = this;

  int i;
  if(l->flatten){
    memcpy(l->output, net.input, l->outputs*l->batch*sizeof(float));
    if(l->reverse){
      ::flatten(l->output, l->w*l->h, l->c, l->batch, 0);
    }else{
      ::flatten(l->output, l->w*l->h, l->c, l->batch, 1);
    }
  } else if (l->extra) {
    for(i = 0; i < l->batch; ++i){
      copy_cpu(l->inputs, net.input + i*l->inputs, 1, l->output + i*l->outputs, 1);
    }
  } else if (l->reverse){
    reorg_cpu(net.input, l->w, l->h, l->c, l->batch, l->stride, 1, l->output);
  } else {
    reorg_cpu(net.input, l->w, l->h, l->c, l->batch, l->stride, 0, l->output);
  }
}

void reorg_layer::backward(network net){
  reorg_layer* l = this;

    int i;
    if(l->flatten){
        memcpy(net.delta, l->delta, l->outputs*l->batch*sizeof(float));
        if(l->reverse){
          ::flatten(net.delta, l->w*l->h, l->c, l->batch, 1);
        }else{
          ::flatten(net.delta, l->w*l->h, l->c, l->batch, 0);
        }
    } else if(l->reverse){
        reorg_cpu(l->delta, l->w, l->h, l->c, l->batch, l->stride, 0, net.delta);
    } else if (l->extra) {
        for(i = 0; i < l->batch; ++i){
            copy_cpu(l->inputs, l->delta + i*l->outputs, 1, net.delta + i*l->inputs, 1);
        }
    }else{
        reorg_cpu(l->delta, l->w, l->h, l->c, l->batch, l->stride, 1, net.delta);
    }
}
