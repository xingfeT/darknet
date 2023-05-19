#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"


struct shortcut_layer :public layer{
  void forward( network net)  ;
  void backward( network net) ;
  void resize( int w, int h);
};

shortcut_layer* make_shortcut_layer(int batch, int index, int w, int h, int c);



#endif
