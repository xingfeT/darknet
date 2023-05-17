// void save_convolutional_weights_binary(layer l, FILE *fp){
//   layer* l = this;
// #ifdef GPU
//   if(gpu_index >= 0){
//     pull_convolutional_layer(l);
//   }
// #endif
//   binarize_weights(l->weights, l->n, l->c*l->size*l->size, l->binary_weights);
//   int size = l->c*l->size*l->size;
//   int i, j, k;
//   fwrite(l->biases, sizeof(float), l->n, fp);
//   if (l->batch_normalize){
//     fwrite(l->scales, sizeof(float), l->n, fp);
//     fwrite(l->rolling_mean, sizeof(float), l->n, fp);
//     fwrite(l->rolling_variance, sizeof(float), l->n, fp);
//   }
//   for(i = 0; i < l->n; ++i){
//     float mean = l->binary_weights[i*size];
//     if(mean < 0) mean = -mean;
//     fwrite(&mean, sizeof(float), 1, fp);
//     for(j = 0; j < size/8; ++j){
//       int index = i*size + j*8;
//       unsigned char c = 0;
//       for(k = 0; k < 8; ++k){
//         if (j*8 + k >= size) break;
//         if (l->binary_weights[index + k] > 0) c = (c | 1<<k);
//       }
//       fwrite(&c, sizeof(char), 1, fp);
//     }
//   }
// }

// void save_convolutional_weights(layer l, FILE *fp){
//   if(l->binary){
//     //save_convolutional_weights_binary(l, fp);
//     //return;
//   }
// #ifdef GPU
//   if(gpu_index >= 0){
//     pull_convolutional_layer(l);
//   }
// #endif
//   int num = l->nweights;
//   fwrite(l->biases, sizeof(float), l->n, fp);
//   if (l->batch_normalize){
//     fwrite(l->scales, sizeof(float), l->n, fp);
//     fwrite(l->rolling_mean, sizeof(float), l->n, fp);
//     fwrite(l->rolling_variance, sizeof(float), l->n, fp);
//   }
//   fwrite(l->weights, sizeof(float), num, fp);
// }

// void save_batchnorm_weights(layer l, FILE *fp)
// {
// #ifdef GPU
//   if(gpu_index >= 0){
//     pull_batchnorm_layer(l);
//   }
// #endif
//   fwrite(l->scales, sizeof(float), l->c, fp);
//   fwrite(l->rolling_mean, sizeof(float), l->c, fp);
//   fwrite(l->rolling_variance, sizeof(float), l->c, fp);
// }

// void save_connected_weights(layer l, FILE *fp)
// {
// #ifdef GPU
//   if(gpu_index >= 0){
//     pull_connected_layer(l);
//   }
// #endif
//   fwrite(l->biases, sizeof(float), l->outputs, fp);
//   fwrite(l->weights, sizeof(float), l->outputs*l->inputs, fp);
//   if (l->batch_normalize){
//     fwrite(l->scales, sizeof(float), l->outputs, fp);
//     fwrite(l->rolling_mean, sizeof(float), l->outputs, fp);
//     fwrite(l->rolling_variance, sizeof(float), l->outputs, fp);
//   }
// }


void save_weights_upto(network *net, char *filename, int cutoff){
#ifdef GPU
  if(net->gpu_index >= 0){
    cuda_set_device(net->gpu_index);
  }
#endif
  fprintf(stderr, "Saving weights to %s\n", filename);
  FILE *fp = fopen(filename, "wb");
  if(!fp) file_error(filename);

  int major = 0;
  int minor = 2;
  int revision = 0;
  fwrite(&major, sizeof(int), 1, fp);
  fwrite(&minor, sizeof(int), 1, fp);
  fwrite(&revision, sizeof(int), 1, fp);
  fwrite(net->seen, sizeof(size_t), 1, fp);

  int i;
  for(i = 0; i < net->n && i < cutoff; ++i){
    layer* l = net->layers[i];
    if (l->dontsave) continue;
    if(l->type == CONVOLUTIONAL || l->type == DECONVOLUTIONAL){
      save_convolutional_weights(l, fp);
    } if(l->type == CONNECTED){
      save_connected_weights(l, fp);
    } if(l->type == BATCHNORM){
      save_batchnorm_weights(l, fp);
    } if(l->type == RNN){
      save_connected_weights(*(l->input_layer), fp);
      save_connected_weights(*(l->self_layer), fp);
      save_connected_weights(*(l->output_layer), fp);
    } if (l->type == LSTM) {
      save_connected_weights(*(l->wi), fp);
      save_connected_weights(*(l->wf), fp);
      save_connected_weights(*(l->wo), fp);
      save_connected_weights(*(l->wg), fp);
      save_connected_weights(*(l->ui), fp);
      save_connected_weights(*(l->uf), fp);
      save_connected_weights(*(l->uo), fp);
      save_connected_weights(*(l->ug), fp);
    } if (l->type == GRU) {
      if(1){
        save_connected_weights(*(l->wz), fp);
        save_connected_weights(*(l->wr), fp);
        save_connected_weights(*(l->wh), fp);
        save_connected_weights(*(l->uz), fp);
        save_connected_weights(*(l->ur), fp);
        save_connected_weights(*(l->uh), fp);
      }else{
        save_connected_weights(*(l->reset_layer), fp);
        save_connected_weights(*(l->update_layer), fp);
        save_connected_weights(*(l->state_layer), fp);
      }
    }  if(l->type == CRNN){
      save_convolutional_weights(*(l->input_layer), fp);
      save_convolutional_weights(*(l->self_layer), fp);
      save_convolutional_weights(*(l->output_layer), fp);
    } if(l->type == LOCAL){
#ifdef GPU
      if(gpu_index >= 0){
        pull_local_layer(l);
      }
#endif
      int locations = l->out_w*l->out_h;
      int size = l->size*l->size*l->c*l->n*locations;
      fwrite(l->biases, sizeof(float), l->outputs, fp);
      fwrite(l->weights, sizeof(float), size, fp);
    }
  }
  fclose(fp);
}

void save_weights(network *net, char *filename){
  save_weights_upto(net, filename, net->n);
}


void load_connected_weights(layer l, FILE *fp, int transpose){
  fread(l->biases, sizeof(float), l->outputs, fp);
  fread(l->weights, sizeof(float), l->outputs*l->inputs, fp);
  if(transpose){
    transpose_matrix(l->weights, l->inputs, l->outputs);
  }
  //printf("Biases: %f mean %f variance\n", mean_array(l->biases, l->outputs), variance_array(l->biases, l->outputs));
  //printf("Weights: %f mean %f variance\n", mean_array(l->weights, l->outputs*l->inputs), variance_array(l->weights, l->outputs*l->inputs));
  if (l->batch_normalize && (!l->dontloadscales)){
    fread(l->scales, sizeof(float), l->outputs, fp);
    fread(l->rolling_mean, sizeof(float), l->outputs, fp);
    fread(l->rolling_variance, sizeof(float), l->outputs, fp);
    //printf("Scales: %f mean %f variance\n", mean_array(l->scales, l->outputs), variance_array(l->scales, l->outputs));
    //printf("rolling_mean: %f mean %f variance\n", mean_array(l->rolling_mean, l->outputs), variance_array(l->rolling_mean, l->outputs));
    //printf("rolling_variance: %f mean %f variance\n", mean_array(l->rolling_variance, l->outputs), variance_array(l->rolling_variance, l->outputs));
  }
#ifdef GPU
  if(gpu_index >= 0){
    push_connected_layer(l);
  }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
  fread(l->scales, sizeof(float), l->c, fp);
  fread(l->rolling_mean, sizeof(float), l->c, fp);
  fread(l->rolling_variance, sizeof(float), l->c, fp);
#ifdef GPU
  if(gpu_index >= 0){
    push_batchnorm_layer(l);
  }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp){
  fread(l->biases, sizeof(float), l->n, fp);
  if (l->batch_normalize && (!l->dontloadscales)){
    fread(l->scales, sizeof(float), l->n, fp);
    fread(l->rolling_mean, sizeof(float), l->n, fp);
    fread(l->rolling_variance, sizeof(float), l->n, fp);
  }
  int size = l->c*l->size*l->size;
  int i, j, k;
  for(i = 0; i < l->n; ++i){
    float mean = 0;
    fread(&mean, sizeof(float), 1, fp);
    for(j = 0; j < size/8; ++j){
      int index = i*size + j*8;
      unsigned char c = 0;
      fread(&c, sizeof(char), 1, fp);
      for(k = 0; k < 8; ++k){
        if (j*8 + k >= size) break;
        l->weights[index + k] = (c & 1<<k) ? mean : -mean;
      }
    }
  }
#ifdef GPU
  if(gpu_index >= 0){
    push_convolutional_layer(l);
  }
#endif
}

void load_convolutional_weights(layer l, FILE *fp){
  if(l->binary){
    //load_convolutional_weights_binary(l, fp);
    //return;
  }
  if(l->numload) l->n = l->numload;
  int num = l->c/l->groups*l->n*l->size*l->size;
  fread(l->biases, sizeof(float), l->n, fp);
  if (l->batch_normalize && (!l->dontloadscales)){
    fread(l->scales, sizeof(float), l->n, fp);
    fread(l->rolling_mean, sizeof(float), l->n, fp);
    fread(l->rolling_variance, sizeof(float), l->n, fp);
    if(0){
      int i;
      for(i = 0; i < l->n; ++i){
        printf("%g, ", l->rolling_mean[i]);
      }
      printf("\n");
      for(i = 0; i < l->n; ++i){
        printf("%g, ", l->rolling_variance[i]);
      }
      printf("\n");
    }
    if(0){
      fill_cpu(l->n, 0, l->rolling_mean, 1);
      fill_cpu(l->n, 0, l->rolling_variance, 1);
    }
    if(0){
      int i;
      for(i = 0; i < l->n; ++i){
        printf("%g, ", l->rolling_mean[i]);
      }
      printf("\n");
      for(i = 0; i < l->n; ++i){
        printf("%g, ", l->rolling_variance[i]);
      }
      printf("\n");
    }
  }
  fread(l->weights, sizeof(float), num, fp);
  //if(l->c == 3) scal_cpu(num, 1./256, l->weights, 1);
  if (l->flipped) {
    transpose_matrix(l->weights, l->c*l->size*l->size, l->n);
  }
  //if (l->binary) binarize_weights(l->weights, l->n, l->c*l->size*l->size, l->weights);
#ifdef GPU
  if(gpu_index >= 0){
    push_convolutional_layer(l);
  }
#endif
}


void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l->dontload) continue;
        if(l->type == CONVOLUTIONAL || l->type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l->type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l->type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l->type == CRNN){
            load_convolutional_weights(*(l->input_layer), fp);
            load_convolutional_weights(*(l->self_layer), fp);
            load_convolutional_weights(*(l->output_layer), fp);
        }
        if(l->type == RNN){
            load_connected_weights(*(l->input_layer), fp, transpose);
            load_connected_weights(*(l->self_layer), fp, transpose);
            load_connected_weights(*(l->output_layer), fp, transpose);
        }
        if (l->type == LSTM) {
            load_connected_weights(*(l->wi), fp, transpose);
            load_connected_weights(*(l->wf), fp, transpose);
            load_connected_weights(*(l->wo), fp, transpose);
            load_connected_weights(*(l->wg), fp, transpose);
            load_connected_weights(*(l->ui), fp, transpose);
            load_connected_weights(*(l->uf), fp, transpose);
            load_connected_weights(*(l->uo), fp, transpose);
            load_connected_weights(*(l->ug), fp, transpose);
        }
        if (l->type == GRU) {
            if(1){
                load_connected_weights(*(l->wz), fp, transpose);
                load_connected_weights(*(l->wr), fp, transpose);
                load_connected_weights(*(l->wh), fp, transpose);
                load_connected_weights(*(l->uz), fp, transpose);
                load_connected_weights(*(l->ur), fp, transpose);
                load_connected_weights(*(l->uh), fp, transpose);
            }else{
                load_connected_weights(*(l->reset_layer), fp, transpose);
                load_connected_weights(*(l->update_layer), fp, transpose);
                load_connected_weights(*(l->state_layer), fp, transpose);
            }
        }
        if(l->type == LOCAL){
            int locations = l->out_w*l->out_h;
            int size = l->size*l->size*l->c*l->n*locations;
            fread(l->biases, sizeof(float), l->outputs, fp);
            fread(l->weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}
