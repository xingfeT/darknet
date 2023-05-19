#pragma once 
#include "image.h"
#include "layer.h"
#include "network.h"

struct crop_layer :public layer{
  image get_crop_image(crop_layer l);
  
  void resize(int w, int h);
  void forward(struct network);
  void backward(struct network);
  void update(update_args);
};



crop_layer* make_crop_layer(int batch, int h, int w, int c,
                            int crop_height, int crop_width,
                            int flip, float angle,
                            float saturation, float exposure);


