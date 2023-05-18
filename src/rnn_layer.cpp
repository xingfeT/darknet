#include "rnn_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void rnn_layer::increment_layer(int steps){
  rnn_layer * l = this;
  
  int num = l->outputs*l->batch*steps;
  l->output += num;
  l->delta += num;
  l->x += num;
  l->x_norm += num;
}

rnn_layer* make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam){
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    rnn_layer* l = new rnn_layer();
    
    l->batch = batch;
    l->type = RNN;
    l->steps = steps;
    l->inputs = inputs;

    l->state = (float*)calloc(batch*outputs*2, sizeof(float));
    l->prev_state =  l->state+(batch*outputs);
    

    l->input_layer = make_connected_layer(batch*steps, inputs, outputs, activation, batch_normalize, adam);malloc(sizeof(layer));
    l->input_layer->batch = batch;
    
    l->self_layer = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
    l->self_layer->batch = batch;

    l->output_layer = make_connected_layer(batch*steps, outputs, outputs, activation, batch_normalize, adam);malloc(sizeof(layer));
    l->output_layer->batch = batch;

    l->outputs = outputs;
    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    return l;
}

void rnn_layer::update(update_args a){
  this->input_layer->update(a);
  this->self_layer->update(a);
  this->output_layer->update(a);
}

void rnn_layer::forward(network net){
    network s = net;
    s.train = net.train;
    int i;

    fill_cpu(this->outputs * this->batch * this->steps, 0, output_layer->delta, 1);
    fill_cpu(this->outputs * this->batch * this->steps, 0, self_layer->delta, 1);
    fill_cpu(this->outputs * this->batch * this->steps, 0, input_layer->delta, 1);
    
    if(net.train) fill_cpu(this->outputs * this->batch, 0, this->state, 1);

    for (i = 0; i < this->steps; ++i) {
        s.input = net.input;
        input_layer->forward(s);

        s.input = this->state;
        self_layer->forward(s);

        float *old_state = this->state;
        if(net.train) this->state += this->outputs*this->batch;
        if(this->shortcut){
            copy_cpu(this->outputs * this->batch, old_state, 1, this->state, 1);
        }else{
            fill_cpu(this->outputs * this->batch, 0, this->state, 1);
        }
        axpy_cpu(this->outputs * this->batch, 1, input_layer->output, 1, this->state, 1);
        axpy_cpu(this->outputs * this->batch, 1, self_layer->output, 1, this->state, 1);

        s.input = this->state;
        output_layer->forward(s);

        net.input += this->inputs*this->batch;
        this->increment_layer(1);
        this->increment_layer(1);
        this->increment_layer(1);
    }
}

void rnn_layer::backward(network net){
    network s = net;
    s.train = net.train;
    int i;


    // increment_layer(&input_layer, this->steps-1);
    // increment_layer(&self_layer, this->steps-1);
    // increment_layer(&output_layer, this->steps-1);

    this->state += this->outputs*this->batch*this->steps;
    for (i = this->steps-1; i >= 0; --i) {
        copy_cpu(this->outputs * this->batch, input_layer->output, 1, this->state, 1);
        axpy_cpu(this->outputs * this->batch, 1, self_layer->output, 1, this->state, 1);

        s.input = this->state;
        s.delta = self_layer->delta;
        output_layer->backward(s);

        this->state -= this->outputs*this->batch;
        /*
           if(i > 0){
           copy_cpu(this->outputs * this->batch, input_layer.output - this->outputs*this->batch, 1, this->state, 1);
           axpy_cpu(this->outputs * this->batch, 1, self_layer.output - this->outputs*this->batch, 1, this->state, 1);
           }else{
           fill_cpu(this->outputs * this->batch, 0, this->state, 1);
           }
         */

        s.input = this->state;
        s.delta = self_layer->delta - this->outputs*this->batch;
        if (i == 0) s.delta = 0;
        self_layer->backward(s);

        copy_cpu(this->outputs*this->batch, self_layer->delta, 1, input_layer->delta, 1);
        if (i > 0 && this->shortcut) {
          axpy_cpu(this->outputs*this->batch, 1, self_layer->delta, 1,
                   self_layer->delta - this->outputs*this->batch, 1);
        }
        
        s.input = net.input + i*this->inputs*this->batch;
        if(net.delta) s.delta = net.delta + i*this->inputs*this->batch;
        else s.delta = 0;
        input_layer->backward( s);

        this->increment_layer( -1);
        this->increment_layer( -1);
        this->increment_layer( -1);
    }
}

