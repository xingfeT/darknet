#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"


COST_TYPE get_cost_type(char *s);
const char *get_cost_string(COST_TYPE a);

struct cost_layer :public layer{
  void forward( network net);
  void backward( network net);
  void update(update_args){}
  void resize( int inputs);
};

cost_layer* make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);


#endif
