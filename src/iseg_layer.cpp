#include "iseg_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

iseg_layer* make_iseg_layer(int batch, int w, int h, int classes, int ids){
  iseg_layer* l = new iseg_layer();
  
    l->type = ISEG;

    l->h = h;
    l->w = w;
    l->c = classes + ids;

    l->out_w = l->w;
    l->out_h = l->h;
    l->out_c = l->c;
    
    l->classes = classes;
    l->batch = batch;
    l->extra = ids;
    l->cost = (float*)calloc(1, sizeof(float));
    l->outputs = h*w*l->c;
    l->inputs = l->outputs;
    
    l->truths = 90*(l->w*l->h+1);
    
    l->delta = (float*)calloc(batch*l->outputs, sizeof(float));
    
    l->output = (float*)calloc(batch*l->outputs, sizeof(float));

    l->counts = (int*)calloc(90, sizeof(int));
    l->sums = (float**)calloc(90, sizeof(float*));
    if(ids){
      for(int i = 0; i < 90; ++i){
        l->sums[i] = (float*)calloc(ids, sizeof(float));
      }
    }


    fprintf(stderr, "iseg\n");
    srand(0);

    return l;
}

void iseg_layer::resize(int w, int h){
  iseg_layer* l = this;
  
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->c;
    l->inputs = l->outputs;

    l->output = (float*)realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = (float*)realloc(l->delta, l->batch*l->outputs*sizeof(float));
}

void iseg_layer::forward(network net){
  iseg_layer* l = this;
  

    double time = what_time_is_it_now();
    int i,b,j,k;
    int ids = l->extra;
    memcpy(l->output, net.input, l->outputs*l->batch*sizeof(float));
    memset(l->delta, 0, l->outputs * l->batch * sizeof(float));


    for (b = 0; b < l->batch; ++b){
        int index = b*l->outputs;
        activate_array(l->output + index, l->classes*l->w*l->h, LOGISTIC);
    }


    for (b = 0; b < l->batch; ++b){
        // a priori, each pixel has no class
        for(i = 0; i < l->classes; ++i){
            for(k = 0; k < l->w*l->h; ++k){
                int index = b*l->outputs + i*l->w*l->h + k;
                l->delta[index] = 0 - l->output[index];
            }
        }

        // a priori, embedding should be small magnitude
        for(i = 0; i < ids; ++i){
            for(k = 0; k < l->w*l->h; ++k){
                int index = b*l->outputs + (i+l->classes)*l->w*l->h + k;
                l->delta[index] = .1 * (0 - l->output[index]);
            }
        }


        memset(l->counts, 0, 90*sizeof(int));
        for(i = 0; i < 90; ++i){
            fill_cpu(ids, 0, l->sums[i], 1);
            
            int c = net.truth[b*l->truths + i*(l->w*l->h+1)];
            if(c < 0) break;
            // add up metric embeddings for each instance
            for(k = 0; k < l->w*l->h; ++k){
                int index = b*l->outputs + c*l->w*l->h + k;
                float v = net.truth[b*l->truths + i*(l->w*l->h + 1) + 1 + k];
                if(v){
                    l->delta[index] = v - l->output[index];
                    axpy_cpu(ids, 1, l->output + b*l->outputs + l->classes*l->w*l->h + k, l->w*l->h, l->sums[i], 1);
                    ++l->counts[i];
                }
            }
        }

        float *mse = (float *)calloc(90, sizeof(float));
        for(i = 0; i < 90; ++i){
            int c = net.truth[b*l->truths + i*(l->w*l->h+1)];
            if(c < 0) break;
            for(k = 0; k < l->w*l->h; ++k){
                float v = net.truth[b*l->truths + i*(l->w*l->h + 1) + 1 + k];
                if(v){
                    int z;
                    float sum = 0;
                    for(z = 0; z < ids; ++z){
                        int index = b*l->outputs + (l->classes + z)*l->w*l->h + k;
                        sum += pow(l->sums[i][z]/l->counts[i] - l->output[index], 2);
                    }
                    mse[i] += sum;
                }
            }
            mse[i] /= l->counts[i];
        }

        // Calculate average embedding
        for(i = 0; i < 90; ++i){
            if(!l->counts[i]) continue;
            scal_cpu(ids, 1.f/l->counts[i], l->sums[i], 1);
            if(b == 0 && net.gpu_index == 0){
                printf("%4d, %6.3f, ", l->counts[i], mse[i]);
                for(j = 0; j < ids; ++j){
                    printf("%6.3f,", l->sums[i][j]);
                }
                printf("\n");
            }
        }
        free(mse);

        // Calculate embedding loss
        for(i = 0; i < 90; ++i){
            if(!l->counts[i]) continue;
            for(k = 0; k < l->w*l->h; ++k){
                float v = net.truth[b*l->truths + i*(l->w*l->h + 1) + 1 + k];
                if(v){
                    for(j = 0; j < 90; ++j){
                        if(!l->counts[j])continue;
                        int z;
                        for(z = 0; z < ids; ++z){
                            int index = b*l->outputs + (l->classes + z)*l->w*l->h + k;
                            float diff = l->sums[j][z] - l->output[index];
                            if (j == i) l->delta[index] +=   diff < 0? -.1 : .1;
                            else        l->delta[index] += -(diff < 0? -.1 : .1);
                        }
                    }
                }
            }
        }

        for(i = 0; i < ids; ++i){
            for(k = 0; k < l->w*l->h; ++k){
                int index = b*l->outputs + (i+l->classes)*l->w*l->h + k;
                l->delta[index] *= .01;
            }
        }
    }

    *(l->cost) = pow(mag_array(l->delta, l->outputs * l->batch), 2);
    printf("took %lf sec\n", what_time_is_it_now() - time);
}

void iseg_layer::backward(network net){
  iseg_layer* l = this;
  
  axpy_cpu(l->batch*l->inputs, 1, l->delta, 1, net.delta, 1);
}


