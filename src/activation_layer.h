#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

struct activation_layer: public layer{
  void forward(network net){
    activation_layer * l = this;
    copy_cpu(l->outputs*l->batch, net.input, 1, l->output, 1);
    activate_array(l->output, l->outputs*l->batch, l->activation);
  }

  void backward(network net){
    activation_layer * l = this;

    gradient_array(l->output, l->outputs*l->batch, l->activation, l->delta);
    copy_cpu(l->outputs*l->batch, l->delta, 1, net.delta, 1);
  }
};


activation_layer* make_activation_layer(int batch, int inputs,
                             ACTIVATION activation);


#endif

