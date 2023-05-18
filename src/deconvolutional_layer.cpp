#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>


size_t deconvolutional_layer::workspaceSize(){
  return (size_t)h*w*size*size*n*sizeof(float);
}

void deconvolutional_layer::bilinear_init(){
    int i,j,f;
    float center = (size-1) / 2.;
    
    for(f = 0; f < n; ++f){
        for(j = 0; j < size; ++j){
            for(i = 0; i < size; ++i){
                float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
                int c = f%c;
                int ind = f*size*size*c + c*size*size + j*size + i;
                weights[ind] = val;
            }
        }
    }
}


deconvolutional_layer* make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size,
                                  int stride, int padding, ACTIVATION activation,
                                  int batch_normalize, int adam){
    int i;
    deconvolutional_layer* l = new deconvolutional_layer();
    l->type = DECONVOLUTIONAL;

    l->h = h;
    l->w = w;
    l->c = c;
    l->n = n;
    l->batch = batch;
    l->stride = stride;
    l->size = size;

    l->nweights = c*n*size*size;
    l->nbiases = n;

    l->weights = (float*)calloc(2*(c*n*size*size+n), sizeof(float));
    l->weight_updates = l->weights+(c*n*size*size);

    l->biases = l->weight_updates+(c*n*size*size);
    l->bias_updates = l->biases+n;
    
    //float scale = n/(size*size*c);
    //printf("scale: %f\n", scale);
    float scale = .02;
    for(i = 0; i < c*n*size*size; ++i) l->weights[i] = scale*rand_normal();
    //bilinear_init(l);
    for(i = 0; i < n; ++i){
        l->biases[i] = 0;
    }
    l->pad = padding;

    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;
    l->out_c = n;
    l->outputs = l->out_w * l->out_h * l->out_c;
    l->inputs = l->w * l->h * l->c;

    scal_cpu(l->nweights, (float)l->out_w*l->out_h/(l->w*l->h), l->weights, 1);

    l->output = (float*)calloc(2*l->batch*l->outputs, sizeof(float));
    l->delta  = l->output + l->batch*l->outputs;
    
    l->batch_normalize = batch_normalize;

    if(batch_normalize){
      l->scales = (float*)calloc(n*6+2*(l->batch*l->outputs), sizeof(float));
        l->scale_updates = l->scales +n;
        for(int i = 0; i < n; ++i){
            l->scales[i] = 1;
        }

        l->mean = l->scale_updates+n;
        l->variance = l->mean +n;

        l->mean_delta = l->variance + n;
        l->variance_delta = l->mean_delta +n;

        l->rolling_mean = l->variance_delta +n;
        l->rolling_variance = l->rolling_mean +n;
        
        l->x = l->rolling_variance+n;
        l->x_norm = l->x + l->batch*l->outputs;
    }
    if(adam){
      float * tmp  = (float*)calloc(2*c*n*size*size+2*n, sizeof(float));
      l->m = tmp;
      l->v = tmp + c*n*size*size;
      l->bias_m =  l->v + c*n*size*size;
      l->scale_m = l->bias_m +n;
      l->bias_v = l->scale_m +n;
      l->scale_v = l->bias_v+n;
    }

#ifdef GPU
    l->forward_gpu = forward_deconvolutional_layer_gpu;
    l->backward_gpu = backward_deconvolutional_layer_gpu;
    l->update_gpu = update_deconvolutional_layer_gpu;

    if(gpu_index >= 0){

        if (adam) {
            l->m_gpu = cuda_make_array(l->m, c*n*size*size);
            l->v_gpu = cuda_make_array(l->v, c*n*size*size);
            l->bias_m_gpu = cuda_make_array(l->bias_m, n);
            l->bias_v_gpu = cuda_make_array(l->bias_v, n);
            l->scale_m_gpu = cuda_make_array(l->scale_m, n);
            l->scale_v_gpu = cuda_make_array(l->scale_v, n);
        }
        l->weights_gpu = cuda_make_array(l->weights, c*n*size*size);
        l->weight_updates_gpu = cuda_make_array(l->weight_updates, c*n*size*size);

        l->biases_gpu = cuda_make_array(l->biases, n);
        l->bias_updates_gpu = cuda_make_array(l->bias_updates, n);

        l->delta_gpu = cuda_make_array(l->delta, l->batch*l->out_h*l->out_w*n);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->out_h*l->out_w*n);

        if(batch_normalize){
            l->mean_gpu = cuda_make_array(0, n);
            l->variance_gpu = cuda_make_array(0, n);

            l->rolling_mean_gpu = cuda_make_array(0, n);
            l->rolling_variance_gpu = cuda_make_array(0, n);

            l->mean_delta_gpu = cuda_make_array(0, n);
            l->variance_delta_gpu = cuda_make_array(0, n);

            l->scales_gpu = cuda_make_array(l->scales, n);
            l->scale_updates_gpu = cuda_make_array(0, n);

            l->x_gpu = cuda_make_array(0, l->batch*l->out_h*l->out_w*n);
            l->x_norm_gpu = cuda_make_array(0, l->batch*l->out_h*l->out_w*n);
        }
    }
    #ifdef CUDNN
        cudnnCreateTensorDescriptor(&l->dstTensorDesc);
        cudnnCreateTensorDescriptor(&l->normTensorDesc);
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    #endif
#endif

    l->activation = activation;
    l->workspace_size = l->workspaceSize();

    fprintf(stderr, "deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l->out_w, l->out_h, l->out_c);

    return l;
}

void deconvolutional_layer::denormalize(){
    int i, j;
    for(i = 0; i < n; ++i){
      float scale ;//= scales[i]/sqrt(rolling_variance[i] + 0.00001);
      for(j = 0; j < c*size*size; ++j){
        weights[i*c*size*size + j] *= scale;
      }
      biases[i] -= rolling_mean[i] * scale;
      scales[i] = 1;
      rolling_mean[i] = 0;
      rolling_variance[i] = 1;
    }
}

void deconvolutional_layer::resize(int h, int w){
    this->h = h;
    this->w = w;
    this->out_h = (this->h - 1) * this->stride + this->size - 2*this->pad;
    this->out_w = (this->w - 1) * this->stride + this->size - 2*this->pad;

    this->outputs = this->out_h * this->out_w * this->out_c;
    this->inputs = this->w * this->h * this->c;

    this->output = (float*)realloc(this->output, this->batch*this->outputs*sizeof(float));
    this->delta  = (float*)realloc(this->delta,  this->batch*this->outputs*sizeof(float));
    
    if(this->batch_normalize){
      this->x = (float*)realloc(this->x, this->batch*this->outputs*sizeof(float));
      this->x_norm  = (float*)realloc(this->x_norm, this->batch*this->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(this->delta_gpu);
    cuda_free(this->output_gpu);

    this->delta_gpu =  cuda_make_array(this->delta,  this->batch*this->outputs);
    this->output_gpu = cuda_make_array(this->output, this->batch*this->outputs);

    if(this->batch_normalize){
        cuda_free(this->x_gpu);
        cuda_free(this->x_norm_gpu);

        this->x_gpu = cuda_make_array(this->output, this->batch*this->outputs);
        this->x_norm_gpu = cuda_make_array(this->output, this->batch*this->outputs);
    }
    #ifdef CUDNN
        cudnnSetTensor4dDescriptor(this->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->batch, this->out_c, this->out_h, this->out_w); 
        cudnnSetTensor4dDescriptor(this->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, this->out_c, 1, 1); 
    #endif
#endif
    this->workspace_size = this->workspaceSize();
}

void deconvolutional_layer::forward(network net){
    int i;

    int m = size*size*n;
    int n = h*w;
    int k = c;

    fill_cpu(outputs*batch, 0, output, 1);

    for(i = 0; i < batch; ++i){
        float *a = weights;
        float *b = net.input + i*c*h*w;
        float *c = net.workspace;

        gemm_cpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_cpu(net.workspace, out_c, out_h, out_w, size, stride, pad, output+i*outputs);
    }
    if (batch_normalize) {
      //forward_batchnorm_layer(l, net);
    } else {
        add_bias(output, biases, batch, n, out_w*out_h);
    }
    activate_array(output, batch*n*out_w*out_h, activation);
}

void deconvolutional_layer::backward(network net){
    int i;

    gradient_array(output, outputs*batch, activation, delta);

    if(batch_normalize){
      //backward_batchnorm_layer(l, net);
    } else {
      backward_bias(bias_updates, delta, batch, n, out_w*out_h);
    }

    //if(net.delta) memset(net.delta, 0, batch*h*w*c*sizeof(float));

    for(i = 0; i < batch; ++i){
        int m = this->c;
        int n = size*size*n;
        int k = h*w;

        float *a = net.input + i*m*k;
        float *b = net.workspace;
        float *c = weight_updates;

        im2col_cpu(delta + i*outputs, out_c, out_h, out_w, 
                size, stride, pad, b);
        gemm_cpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(net.delta){
            int m = this->c;
            int n = h*w;
            int k = size*size*n;

            float *a = weights;
            float *b = net.workspace;
            float *c = net.delta + i*n*m;

            gemm_cpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void deconvolutional_layer::update(update_args a){
    float learning_rate = a.learning_rate*learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = size*size*c*n;
    axpy_cpu(n, learning_rate/batch, bias_updates, 1, biases, 1);
    scal_cpu(n, momentum, bias_updates, 1);

    if(scales){
        axpy_cpu(n, learning_rate/batch, scale_updates, 1, scales, 1);
        scal_cpu(n, momentum, scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, weights, 1, weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, weight_updates, 1, weights, 1);
    
    scal_cpu(size, momentum, weight_updates, 1);
}



