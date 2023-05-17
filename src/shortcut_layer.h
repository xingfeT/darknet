#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"


struct shortcut_layer :public layer{
  void forward( network net) const ;
  void backward( network net) const ;
  void resize( int w, int h);
};
shortcut_layer* make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);


#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
#endif

#endif
