#include "gru_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include <initializer_list>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gru_layer::increment_layer(int steps){
  gru_layer * l = this;
  
    int num = l->outputs*l->batch*steps;
    l->output += num;
    
    l->delta += num;
    l->x += num;
    l->x_norm += num;
}

gru_layer* make_gru_layer(int batch, int inputs, int outputs, int steps,
                      int batch_normalize, int adam){
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    gru_layer* l = new gru_layer();
    
    l->batch = batch;
    l->type = GRU;
    l->steps = steps;
    l->inputs = inputs;

    l->uz = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l->uz->batch = batch;

    l->wz = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l->wz->batch = batch;

    l->ur = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l->ur->batch = batch;

    l->wr = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l->wr->batch = batch;

    l->uh = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
    l->uh->batch = batch;

    l->wh = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize, adam);
    l->wh->batch = batch;

    l->batch_normalize = batch_normalize;


    l->outputs = outputs;

    l->output = (float*)calloc(outputs*batch*steps, sizeof(float));
    l->delta = (float*)calloc(outputs*batch*steps, sizeof(float));
    l->state = (float*)calloc(outputs*batch, sizeof(float));
    l->prev_state = (float*)calloc(outputs*batch, sizeof(float));
    l->forgot_state = (float*)calloc(outputs*batch, sizeof(float));
    l->forgot_delta = (float*)calloc(outputs*batch, sizeof(float));

    l->r_cpu = (float*)calloc(outputs*batch, sizeof(float));
    l->z_cpu = (float*)calloc(outputs*batch, sizeof(float));
    l->h_cpu = (float*)calloc(outputs*batch, sizeof(float));

    return l;
}

void gru_layer::update(update_args a){
  for(auto l : {ur, uz, uh , wr, wz, wh}){
    l->update(a);
  }
}

void gru_layer::forward(network net){
  gru_layer* l = this;
  
    network s = net;
    s.train = net.train;
    int i;
    for(auto u : {ur, uz, uh , wr, wz, wh}){
      fill_cpu(l->outputs * l->batch * l->steps, 0, u->delta, 1);
    }
    
    if(net.train) {
      fill_cpu(l->outputs * l->batch * l->steps, 0, l->delta, 1);
      copy_cpu(l->outputs*l->batch, l->state, 1, l->prev_state, 1);
    }

    for (i = 0; i < l->steps; ++i) {
        s.input = l->state;
        wz->forward(s);
        wr->forward(s);

        s.input = net.input;
        uz->forward( s);
        ur->forward( s);
        uh->forward( s);


        copy_cpu(l->outputs*l->batch, uz->output, 1, l->z_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, wz->output, 1, l->z_cpu, 1);

        copy_cpu(l->outputs*l->batch, ur->output, 1, l->r_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, wr->output, 1, l->r_cpu, 1);

        activate_array(l->z_cpu, l->outputs*l->batch, LOGISTIC);
        activate_array(l->r_cpu, l->outputs*l->batch, LOGISTIC);

        copy_cpu(l->outputs*l->batch, l->state, 1, l->forgot_state, 1);
        mul_cpu(l->outputs*l->batch, l->r_cpu, 1, l->forgot_state, 1);

        s.input = l->forgot_state;
        wh->forward(s);

        copy_cpu(l->outputs*l->batch, uh->output, 1, l->h_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, wh->output, 1, l->h_cpu, 1);

        if(l->tanh){
            activate_array(l->h_cpu, l->outputs*l->batch, TANH);
        } else {
            activate_array(l->h_cpu, l->outputs*l->batch, LOGISTIC);
        }

        weighted_sum_cpu(l->state, l->h_cpu, l->z_cpu, l->outputs*l->batch, l->output);

        copy_cpu(l->outputs*l->batch, l->output, 1, l->state, 1);

        net.input += l->inputs*l->batch;
        l->output += l->outputs*l->batch;
        for(auto u :{uz, ur, uh, wz, wr, wh}){
          u->increment_layer(1);
        }

    }
}

void gru_layer::backward(network net){
}

