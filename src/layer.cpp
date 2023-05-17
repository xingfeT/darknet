#include "layer.h"
#include "cuda.h"

#include <stdlib.h>
// struct CpuLayer :public layer{
//   ~CpuLayer(){
//     if(this->type == DROPOUT){
//       if(this->rand)           free(this->rand);
//       return;
//     }

//     if(this->cweights)           free(this->cweights);
//     if(this->indexes)            free(this->indexes);
//     if(this->input_layers)       free(this->input_layers);
//     if(this->input_sizes)        free(this->input_sizes);
//     if(this->map)                free(this->map);
//     if(this->rand)               free(this->rand);
//     if(this->cost)               free(this->cost);
//     if(this->state)              free(this->state);
//     if(this->prev_state)         free(this->prev_state);
//     if(this->forgot_state)       free(this->forgot_state);
//     if(this->forgot_delta)       free(this->forgot_delta);
//     if(this->state_delta)        free(this->state_delta);
//     if(this->concat)             free(this->concat);
//     if(this->concat_delta)       free(this->concat_delta);
//     if(this->binary_weights)     free(this->binary_weights);
//     if(this->biases)             free(this->biases);
//     if(this->bias_updates)       free(this->bias_updates);
//     if(this->scales)             free(this->scales);
//     if(this->scale_updates)      free(this->scale_updates);
//     if(this->weights)            free(this->weights);
//     if(this->weight_updates)     free(this->weight_updates);
//     if(this->delta)              free(this->delta);
//     if(this->output)             free(this->output);
//     if(this->squared)            free(this->squared);
//     if(this->norms)              free(this->norms);
//     if(this->spatial_mean)       free(this->spatial_mean);
//     if(this->mean)               free(this->mean);
//     if(this->variance)           free(this->variance);
//     if(this->mean_delta)         free(this->mean_delta);
//     if(this->variance_delta)     free(this->variance_delta);
//     if(this->rolling_mean)       free(this->rolling_mean);
//     if(this->rolling_variance)   free(this->rolling_variance);
//     if(this->x)                  free(this->x);
//     if(this->x_norm)             free(this->x_norm);
//     if(this->m)                  free(this->m);
//     if(this->v)                  free(this->v);
//     if(this->z_cpu)              free(this->z_cpu);
//     if(this->r_cpu)              free(this->r_cpu);
//     if(this->h_cpu)              free(this->h_cpu);
//     if(this->binary_input)       free(this->binary_input);

//   }
// };

// struct GpuLayer :public layer{
//   ~GpuLayer(){
//     if(l.type == DROPOUT){
//         if(l.rand_gpu)             cuda_free(l.rand_gpu);
//         return;
//     }
//     if(l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

//     if(l.z_gpu)                   cuda_free(l.z_gpu);
//     if(l.r_gpu)                   cuda_free(l.r_gpu);
//     if(l.h_gpu)                   cuda_free(l.h_gpu);
//     if(l.m_gpu)                   cuda_free(l.m_gpu);
//     if(l.v_gpu)                   cuda_free(l.v_gpu);
//     if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
//     if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
//     if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
//     if(l.state_gpu)               cuda_free(l.state_gpu);
//     if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
//     if(l.gate_gpu)                cuda_free(l.gate_gpu);
//     if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
//     if(l.save_gpu)                cuda_free(l.save_gpu);
//     if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
//     if(l.concat_gpu)              cuda_free(l.concat_gpu);
//     if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
//     if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
//     if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
//     if(l.mean_gpu)                cuda_free(l.mean_gpu);
//     if(l.variance_gpu)            cuda_free(l.variance_gpu);
//     if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
//     if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
//     if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
//     if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
//     if(l.x_gpu)                   cuda_free(l.x_gpu);
//     if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
//     if(l.weights_gpu)             cuda_free(l.weights_gpu);
//     if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
//     if(l.biases_gpu)              cuda_free(l.biases_gpu);
//     if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
//     if(l.scales_gpu)              cuda_free(l.scales_gpu);
//     if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
//     if(l.output_gpu)              cuda_free(l.output_gpu);
//     if(l.delta_gpu)               cuda_free(l.delta_gpu);
//     if(l.rand_gpu)                cuda_free(l.rand_gpu);
//     if(l.squared_gpu)             cuda_free(l.squared_gpu);
//     if(l.norms_gpu)               cuda_free(l.norms_gpu);

// }

//};
