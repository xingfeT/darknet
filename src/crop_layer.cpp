#include "crop_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_crop_image(crop_layer l){
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return float_to_image(w,h,c,l.output);
}

void crop_layer::backward(network net){}

crop_layer* make_crop_layer(int batch, int h, int w, int c,
                            int crop_height, int crop_width,
                            int flip, float angle, float saturation, float exposure){
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer* l = new crop_layer();
    l->type = CROP;
    l->batch = batch;
    l->h = h;
    l->w = w;
    l->c = c;
    l->scale = (float)crop_height / h;
    l->flip = flip;
    l->angle = angle;
    l->saturation = saturation;
    l->exposure = exposure;
    l->out_w = crop_width;
    l->out_h = crop_height;
    l->out_c = c;
    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_w * l->out_h * l->out_c;
    
    l->output = (float*)calloc(l->outputs*batch, sizeof(float));
    return l;
}

void crop_layer::resize(int w, int h){
  crop_layer* l = this;
  
    l->w = w;
    l->h = h;

    l->out_w =  l->scale*w;
    l->out_h =  l->scale*h;

    l->inputs = l->w * l->h * l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
}


void crop_layer::forward(network net){
  crop_layer* l = this;
  
  int i,j,c,b,row,col;
  int index;
  int count = 0;
  int flip = (l->flip && std::rand()%2);
  int dh = std::rand()%(l->h - l->out_h + 1);
  int dw = std::rand()%(l->w - l->out_w + 1);
  float scale = 2;
  float trans = -1;
  if(l->noadjust){
    scale = 1;
    trans = 0;
  }
  if(!net.train){
    flip = 0;
    dh = (l->h - l->out_h)/2;
    dw = (l->w - l->out_w)/2;
  }
  for(b = 0; b < l->batch; ++b){
    for(c = 0; c < l->c; ++c){
      for(i = 0; i < l->out_h; ++i){
        for(j = 0; j < l->out_w; ++j){
          if(flip){
            col = l->w - dw - j - 1;    
          }else{
            col = j + dw;
          }
          row = i + dh;
          index = col+l->w*(row+l->h*(c + l->c*b)); 
          l->output[count++] = net.input[index]*scale + trans;
        }
      }
    }
  }
}

