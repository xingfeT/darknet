#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"


COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);

struct CostLayer :public layer{
  void forward( network net);
  void backward( network net);
  void update(update_args);
  void resize_cost_layer( int inputs);
};
typedef CostLayer cost_layer;


CostLayer* make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);


#ifdef GPU
void forward_cost_layer_gpu(cost_layer l, network net);
void backward_cost_layer_gpu(const cost_layer l, network net);
#endif

#endif
