#include "crnn_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void crnn_layer::increment_layer(int steps){
  crnn_layer* l = this;
  
  int num = l->outputs*l->batch*steps;
  l->output += num;
  l->delta += num;
  l->x += num;
  l->x_norm += num;

}

crnn_layer* make_crnn_layer(int batch, int h, int w, int c,
                            int hidden_filters, int output_filters, int steps,
                            ACTIVATION activation, int batch_normalize){
    fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h,w,c,output_filters);
    batch = batch / steps;
    crnn_layer* l = new crnn_layer();
    
    l->batch = batch;
    l->type = CRNN;
    l->steps = steps;
    
    l->h = h;
    l->w = w;
    l->c = c;
    l->out_h = h;
    l->out_w = w;
    l->out_c = output_filters;
    
    l->inputs = h*w*c;
    l->hidden = h * w * hidden_filters;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->state = (float*)calloc(l->hidden*batch*(steps+1), sizeof(float));

    l->input_layer = make_convolutional_layer(batch*steps, h, w, c, hidden_filters,
                                              1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l->input_layer->batch = batch;

    l->self_layer = make_convolutional_layer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
    l->self_layer->batch = batch;

    l->output_layer = make_convolutional_layer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);

    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    return l;
}

void crnn_layer::update(update_args a){
  this->input_layer->update(a);
  this->self_layer->update(a);
  this->output_layer->update(a);
}

void crnn_layer::forward(network net){
    network s = net;
    s.train = net.train;
    
    int i;
    crnn_layer* l = this;
    
    fill_cpu(l->outputs * l->batch * l->steps, 0, output_layer->delta, 1);
    fill_cpu(l->hidden * l->batch * l->steps, 0, self_layer->delta, 1);
    fill_cpu(l->hidden * l->batch * l->steps, 0, input_layer->delta, 1);
    
    if(net.train) fill_cpu(l->hidden * l->batch, 0, l->state, 1);

    for (i = 0; i < l->steps; ++i) {
        s.input = net.input;
        input_layer->forward(s);

        s.input = l->state;
        self_layer->forward(s);

        float *old_state = l->state;
        if(net.train) l->state += l->hidden*l->batch;
        if(l->shortcut){
            copy_cpu(l->hidden * l->batch, old_state, 1, l->state, 1);
        }else{
            fill_cpu(l->hidden * l->batch, 0, l->state, 1);
        }
        axpy_cpu(l->hidden * l->batch, 1, input_layer->output, 1, l->state, 1);
        axpy_cpu(l->hidden * l->batch, 1, self_layer->output, 1, l->state, 1);

        s.input = l->state;
        output_layer->forward(s);

        net.input += l->inputs*l->batch;
        increment_layer(1);
        increment_layer(1);
        increment_layer(1);
    }
}

void crnn_layer::backward(network net){
    network s = net;
    int i;
    crnn_layer* l = this;
    
    increment_layer(l->steps-1);
    increment_layer(l->steps-1);
    increment_layer(l->steps-1);

    l->state += l->hidden*l->batch*l->steps;
    for (i = l->steps-1; i >= 0; --i) {
        copy_cpu(l->hidden * l->batch, input_layer->output, 1, l->state, 1);
        axpy_cpu(l->hidden * l->batch, 1, self_layer->output, 1, l->state, 1);

        s.input = l->state;
        s.delta = self_layer->delta;
        output_layer->backward(s);

        l->state -= l->hidden*l->batch;
        /*
           if(i > 0){
           copy_cpu(l->hidden * l->batch, input_layer.output - l->hidden*l->batch, 1, l->state, 1);
           axpy_cpu(l->hidden * l->batch, 1, self_layer.output - l->hidden*l->batch, 1, l->state, 1);
           }else{
           fill_cpu(l->hidden * l->batch, 0, l->state, 1);
           }
         */

        s.input = l->state;
        s.delta = self_layer->delta - l->hidden*l->batch;
        if (i == 0) s.delta = 0;
        self_layer->backward(s);

        copy_cpu(l->hidden*l->batch, self_layer->delta, 1, input_layer->delta, 1);
        if (i > 0 && l->shortcut) axpy_cpu(l->hidden*l->batch, 1, self_layer->delta, 1, self_layer->delta - l->hidden*l->batch, 1);
        s.input = net.input + i*l->inputs*l->batch;
        if(net.delta) s.delta = net.delta + i*l->inputs*l->batch;
        else s.delta = 0;
        input_layer->backward(s);

        increment_layer(-1);
        increment_layer(-1);
        increment_layer(-1);
    }
}

