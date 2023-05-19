#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

shortcut_layer* make_shortcut_layer(int batch, int index, int w, int h, int c){
  
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index,
            w,h,c,
            w,h,c);
    shortcut_layer* l = new shortcut_layer();
    l->type = SHORTCUT;
    
    l->batch = batch;

    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
  
    // l->w = w2;
    // l->h = h2;
    // 
    
    l->c = c;
    l->out_c = c;
    l->index = index;

    l->resize(w, h);
    return l;
}

void shortcut_layer::resize(int w, int h){
  shortcut_layer* l  = this;
    
  assert(l->w == l->out_w);
  assert(l->h == l->out_h);
  
  l->w = l->out_w = w;
  l->h = l->out_h = h;
    
  l->outputs = w*h*l->out_c;
  l->inputs = l->outputs;
  l->delta =  (float*)realloc(l->delta, l->outputs*l->batch*sizeof(float));
  l->output = (float*)realloc(l->output, l->outputs*l->batch*sizeof(float));    
}


void shortcut_layer::forward(network net){
  shortcut_layer* l  = this;
  copy_cpu(l->outputs*l->batch, net.input, 1, l->output, 1);
  
  shortcut_cpu(l->batch, l->w, l->h, l->c, net.layers[l->index]->output,
               l->out_w, l->out_h, l->out_c, l->alpha, l->beta, l->output);
  activate_array(l->output, l->outputs*l->batch, l->activation);
}

void shortcut_layer::backward( network net){
  shortcut_layer* l  = this;
  gradient_array(l->output, l->outputs*l->batch, l->activation, l->delta);
  axpy_cpu(l->outputs*l->batch, l->alpha, l->delta, 1, net.delta, 1);
  shortcut_cpu(l->batch, l->out_w, l->out_h, l->out_c, l->delta, l->w, l->h, l->c, 1, l->beta,
               net.layers[l->index]->delta);
}

