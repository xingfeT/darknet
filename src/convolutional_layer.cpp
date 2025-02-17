#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l){
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_layer::out_height(){
  convolutional_layer* l = this;
  return (l->h + 2*l->pad - l->size) / l->stride + 1;
}

int convolutional_layer::out_width(){
  convolutional_layer* l = this;
  return (l->w + 2*l->pad - l->size) / l->stride + 1;
}

image convolutional_layer::get_image(){
  convolutional_layer* l = this;
    return float_to_image(l->out_w,l->out_h,l->out_c,l->output);
}

image convolutional_layer::get_delta(){
  convolutional_layer* l = this;

  return float_to_image(l->out_w,l->out_h,l->out_c,l->delta);
}

size_t convolutional_layer::workspaceSize(){
  return (size_t)this->out_h*this->out_w*this->size*this->size*this->c/this->groups*sizeof(float);
}


convolutional_layer* make_convolutional_layer(int batch, int h, int w, int c, int n,
                                              int groups, int size, int stride, int padding,
                                              ACTIVATION activation, int batch_normalize,
                                              int binary, int xnor, int adam){
    int i;
    convolutional_layer* l = new convolutional_layer();
    l->type = CONVOLUTIONAL;

    l->groups = groups;
    l->h = h;
    l->w = w;
    l->c = c;
    l->n = n;
    l->binary = binary;
    l->xnor = xnor;
    l->batch = batch;
    l->stride = stride;
    l->size = size;
    l->pad = padding;

    l->batch_normalize = batch_normalize;

    l->weights = (float*)calloc(c/groups*n*size*size, sizeof(float));
    l->weight_updates = (float*)calloc(c/groups*n*size*size, sizeof(float));

    l->biases = (float*)calloc(n, sizeof(float));
    l->bias_updates = (float*)calloc(n, sizeof(float));

    l->nweights = c/groups*n*size*size;
    l->nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l->groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l->weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l->nweights; ++i) l->weights[i] = scale*rand_normal();

    int out_w = l->out_width();
    int out_h = l->out_height();

    l->out_h = out_h;
    l->out_w = out_w;
    l->out_c = n;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)calloc(l->batch*l->outputs, sizeof(float));
    l->delta  = (float*)calloc(l->batch*l->outputs, sizeof(float));


    if(binary){
      //todo
      //l->binary_weights = calloc(l->nweights, sizeof(float));
      //l->cweights = calloc(l->nweights, sizeof(char));
      //l->scales = calloc(n, sizeof(float));
    }

    if(xnor){
      //todo
      //l->binary_weights = calloc(l->nweights, sizeof(float));
      //l->binary_input = calloc(l->inputs*l->batch, sizeof(float));
    }

    if(batch_normalize){
      l->batch_normalize_layer = make_batchnorm_layer(1, w,h,n);
      l->x = (float*)calloc(l->batch*l->outputs, sizeof(float));
      l->x_norm = (float*)calloc(l->batch*l->outputs, sizeof(float));
    }

    if(adam){
      //todo
      // l->m = calloc(l->nweights, sizeof(float));
      // l->v = calloc(l->nweights, sizeof(float));
      // l->bias_m = calloc(n, sizeof(float));
      // l->scale_m = calloc(n, sizeof(float));
      // l->bias_v = calloc(n, sizeof(float));
      // l->scale_v = calloc(n, sizeof(float));
    }

    l->activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l->out_w, l->out_h, l->out_c, (2.0 * l->n * l->size*l->size*l->c/l->groups * l->out_h*l->out_w)/1000000000.);

    return l;
}

void convolutional_layer::denormalize(){
  convolutional_layer* l  = this;
  for(int i = 0; i < l->n; ++i){
    float scale;// = l->scales[i]/sqrt(l->rolling_variance[i] + 0.00001);
    for(int j = 0; j < l->c/l->groups*l->size*l->size; ++j){
      l->weights[i*l->c/l->groups*l->size*l->size + j] *= scale;
    }
    l->biases[i] -= l->rolling_mean[i] * scale;
    l->scales[i] = 1;
    l->rolling_mean[i] = 0;
    l->rolling_variance[i] = 1;
  }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l->batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void convolutional_layer::resize(int w, int h){
  convolutional_layer* l = this;

    l->w = w;
    l->h = h;
    int out_w = l->out_width();
    int out_h = l->out_height();

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = (float*)realloc(l->delta,  l->batch*l->outputs*sizeof(float));

    if(l->batch_normalize){
      l->x = (float*)realloc(l->x, l->batch*l->outputs*sizeof(float));
      l->x_norm  = (float*)realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }
    l->workspace_size = l->workspaceSize();
}

void add_bias(float *output, float *biases, int batch, int n, int size){
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size){
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size){
  for(int b = 0; b < batch; ++b){
    for(int i = 0; i < n; ++i){
      bias_updates[i] += sum_array(delta+size*(i+b*n), size);
    }
  }
}

void convolutional_layer::forward(network net){
  convolutional_layer* l = this;

  int i, j;
  fill_cpu(l->outputs*l->batch, 0, l->output, 1);

  if(l->xnor){
    binarize_weights(l->weights, l->n, l->c/l->groups*l->size*l->size, l->binary_weights);
    // swap_binary(&l);
    binarize_cpu(net.input, l->c*l->h*l->w*l->batch, l->binary_input);
    net.input = l->binary_input;
  }

  int m = l->n/l->groups;
  int k = l->size*l->size*l->c/l->groups;
  int n = l->out_w*l->out_h;
  for(i = 0; i < l->batch; ++i){
    for(j = 0; j < l->groups; ++j){
      float *a = l->weights + j*l->nweights/l->groups;
      float *b = net.workspace;
      float *c = l->output + (i*l->groups + j)*n*m;
      float *im =  net.input + (i*l->groups + j)*l->c/l->groups*l->h*l->w;

      if (l->size == 1) {
        b = im;
      } else {
        im2col_cpu(im, l->c/l->groups, l->h, l->w, l->size, l->stride, l->pad, b);
      }
      gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
  }

    if(l->batch_normalize){
      l->batchnorm_layer->forward(net);
    } else {
      add_bias(l->output, l->biases, l->batch, l->n, l->out_h*l->out_w);
    }

    activate_array(l->output, l->outputs*l->batch, l->activation);
    //if(l->binary || l->xnor) swap_binary(&l);
}

void convolutional_layer::backward(network net){
  convolutional_layer* l = this;
    int i, j;
    int m = l->n/l->groups;
    int n = l->size*l->size*l->c/l->groups;
    int k = l->out_w*l->out_h;

    gradient_array(l->output, l->outputs*l->batch, l->activation, l->delta);

    if(l->batch_normalize){
      l->batchnorm_layer->backward(net);
    } else {
        backward_bias(l->bias_updates, l->delta, l->batch, l->n, k);
    }

    for(i = 0; i < l->batch; ++i){
        for(j = 0; j < l->groups; ++j){
            float *a = l->delta + (i*l->groups + j)*m*k;
            float *b = net.workspace;
            float *c = l->weight_updates + j*l->nweights/l->groups;

            float *im  = net.input + (i*l->groups + j)*l->c/l->groups*l->h*l->w;
            float *imd = net.delta + (i*l->groups + j)*l->c/l->groups*l->h*l->w;

            if(l->size == 1){
                b = im;
            } else {
                im2col_cpu(im, l->c/l->groups, l->h, l->w,
                        l->size, l->stride, l->pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l->weights + j*l->nweights/l->groups;
                b = l->delta + (i*l->groups + j)*m*k;
                c = net.workspace;
                if (l->size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l->size != 1) {
                  //col2im_cpu(net.workspace, l->c/l->groups, l->h, l->w, l->size, l->stride, l->pad, imd);
                }
            }
        }
    }
}

void convolutional_layer::update(update_args a){
  convolutional_layer* l = this;

    float learning_rate = a.learning_rate*l->learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l->n, learning_rate/batch, l->bias_updates, 1, l->biases, 1);
    scal_cpu(l->n, momentum, l->bias_updates, 1);

    if(l->scales){
        axpy_cpu(l->n, learning_rate/batch, l->scale_updates, 1, l->scales, 1);
        scal_cpu(l->n, momentum, l->scale_updates, 1);
    }

    axpy_cpu(l->nweights, -decay*batch, l->weights, 1, l->weight_updates, 1);
    axpy_cpu(l->nweights, learning_rate/batch, l->weight_updates, 1, l->weights, 1);
    scal_cpu(l->nweights, momentum, l->weight_updates, 1);
}


image convolutional_layer::get_weight(int i){
  convolutional_layer* l = this;
  int h = l->size;
  int w = l->size;
  int c = l->c/l->groups;
  return float_to_image(w,h,c,l->weights+i*h*w*c);
}

void convolutional_layer::rgbgr_weights(){
  convolutional_layer* l = this;

  for(int i = 0; i < l->n; ++i){
    image im = l->get_weight(i);
    if (im.c == 3) {
      rgbgr_image(im);
    }
  }
}

void convolutional_layer::rescale_weights(float scale, float trans){
  convolutional_layer* l = this;
  int i;
  for(i = 0; i < l->n; ++i){
    image im = l->get_weight(i);
    if (im.c == 3) {
      scale_image(im, scale);
      float sum = sum_array(im.data, im.w*im.h*im.c);
      l->biases[i] += sum*trans;
    }
  }
}

image *convolutional_layer::get_weights(){
  convolutional_layer* l = this;

  image *weights = (image *)calloc(l->n, sizeof(image));
  for(int i = 0; i < l->n; ++i){
    weights[i] = copy_image(l->get_weight(i));
    normalize_image(weights[i]);
    /*
      char buff[256];
      sprintf(buff, "filter%d", i);
      save_image(weights[i], buff);
    */
  }
  //error("hey");
  return weights;
}

image *convolutional_layer::visualize(char *window, image *prev_weights){
  convolutional_layer* l = this;

  image *single_weights = l->get_weights();
  show_images(single_weights, l->n, window);

  image delta = l->get_image();
  image dc = collapse_image_layers(delta, 1);
  char buff[256];
  sprintf(buff, "%s: Output", window);
  //show_image(dc, buff);
  //save_image(dc, buff);
  free_image(dc);
  return single_weights;
}
