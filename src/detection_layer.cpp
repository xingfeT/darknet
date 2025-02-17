#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

detection_layer* make_detection_layer(int batch,
                                      int inputs, int n, int side, int classes, int coords,
                                      int rescore){
  detection_layer* l = new detection_layer();
    l->type = DETECTION;

    l->n = n;
    l->batch = batch;
    l->inputs = inputs;
    l->classes = classes;
    l->coords = coords;
    l->rescore = rescore;
    l->side = side;
    l->w = side;
    l->h = side;
    assert(side*side*((1 + l->coords)*l->n + l->classes) == inputs);
    
    l->cost = (float*)calloc(1, sizeof(float));
    l->outputs = l->inputs;
    l->truths = l->side*l->side*(1+l->coords+l->classes);

    l->output = (float*)calloc(batch*l->outputs, sizeof(float));
    l->delta = (float*)calloc(batch*l->outputs, sizeof(float));


    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void detection_layer::forward(network net){
  detection_layer* l = this;
  
  int locations = l->side*l->side;
  int i,j;
  memcpy(l->output, net.input, l->outputs*l->batch*sizeof(float));
  //if(l->reorg) reorg(l->output, l->w*l->h, size*l->n, l->batch, 1);
  int b;
  
  if (l->softmax){
    for(b = 0; b < l->batch; ++b){
      int index = b*l->inputs;
      for (i = 0; i < locations; ++i) {
        int offset = i*l->classes;
        
        // softmax(l->output + index + offset, l->classes, 1, 1,
        //         l->output + index + offset);
      }
    }
  }
    
  if(net.train){
    float avg_iou = 0;
    float avg_cat = 0;
    float avg_allcat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    *(l->cost) = 0;
    int size = l->inputs * l->batch;
    memset(l->delta, 0, size * sizeof(float));
    for (b = 0; b < l->batch; ++b){
      int index = b*l->inputs;
      for (i = 0; i < locations; ++i) {
        int truth_index = (b*locations + i)*(1+l->coords+l->classes);
        int is_obj = net.truth[truth_index];
        for (j = 0; j < l->n; ++j) {
          int p_index = index + locations*l->classes + i*l->n + j;
          l->delta[p_index] = l->noobject_scale*(0 - l->output[p_index]);
          *(l->cost) += l->noobject_scale*pow(l->output[p_index], 2);
          avg_anyobj += l->output[p_index];
        }

        int best_index = -1;
        float best_iou = 0;
        float best_rmse = 20;

        if (!is_obj){
          continue;
        }

        int class_index = index + i*l->classes;
        for(j = 0; j < l->classes; ++j) {
          l->delta[class_index+j] = l->class_scale * (net.truth[truth_index+1+j] - l->output[class_index+j]);
          *(l->cost) += l->class_scale * pow(net.truth[truth_index+1+j] - l->output[class_index+j], 2);
          if(net.truth[truth_index + 1 + j]) avg_cat += l->output[class_index+j];
          avg_allcat += l->output[class_index+j];
        }

        box truth = float_to_box(net.truth + truth_index + 1 + l->classes, 1);
        truth.x /= l->side;
        truth.y /= l->side;

        for(j = 0; j < l->n; ++j){
          int box_index = index + locations*(l->classes + l->n) + (i*l->n + j) * l->coords;
          box out = float_to_box(l->output + box_index, 1);
          out.x /= l->side;
          out.y /= l->side;

          if (l->sqrt){
            out.w = out.w*out.w;
            out.h = out.h*out.h;
          }

          float iou  = box_iou(out, truth);
          //iou = 0;
          float rmse = box_rmse(out, truth);
          if(best_iou > 0 || iou > 0){
            if(iou > best_iou){
              best_iou = iou;
              best_index = j;
            }
          }else{
            if(rmse < best_rmse){
              best_rmse = rmse;
              best_index = j;
            }
          }
        }

        if(l->forced){
          if(truth.w*truth.h < .1){
            best_index = 1;
          }else{
            best_index = 0;
          }
        }
        if(l->random && *(net.seen) < 64000){
          //best_index = rand()%(l->n);
        }

        int box_index = index + locations*(l->classes + l->n) + (i*l->n + best_index) * l->coords;
        int tbox_index = truth_index + 1 + l->classes;

        box out = float_to_box(l->output + box_index, 1);
        out.x /= l->side;
        out.y /= l->side;
        if (l->sqrt) {
          out.w = out.w*out.w;
          out.h = out.h*out.h;
        }
        float iou  = box_iou(out, truth);

        //printf("%d,", best_index);
        int p_index = index + locations*l->classes + i*l->n + best_index;
        *(l->cost) -= l->noobject_scale * pow(l->output[p_index], 2);
        *(l->cost) += l->object_scale * pow(1-l->output[p_index], 2);
        avg_obj += l->output[p_index];
        l->delta[p_index] = l->object_scale * (1.-l->output[p_index]);

        if(l->rescore){
          l->delta[p_index] = l->object_scale * (iou - l->output[p_index]);
        }

        l->delta[box_index+0] = l->coord_scale*(net.truth[tbox_index + 0] - l->output[box_index + 0]);
        l->delta[box_index+1] = l->coord_scale*(net.truth[tbox_index + 1] - l->output[box_index + 1]);
        l->delta[box_index+2] = l->coord_scale*(net.truth[tbox_index + 2] - l->output[box_index + 2]);
        l->delta[box_index+3] = l->coord_scale*(net.truth[tbox_index + 3] - l->output[box_index + 3]);
        if(l->sqrt){
          l->delta[box_index+2] = l->coord_scale*(std::sqrt(net.truth[tbox_index + 2])
                                                  - l->output[box_index + 2]);
          l->delta[box_index+3] = l->coord_scale*(std::sqrt(net.truth[tbox_index + 3])
                                                  - l->output[box_index + 3]);
        }

        *(l->cost) += pow(1-iou, 2);
        avg_iou += iou;
        ++count;
      }
    }
    
    *(l->cost) = pow(mag_array(l->delta, l->outputs * l->batch), 2);

    printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l->classes), avg_obj/count, avg_anyobj/(l->batch*locations*l->n), count);
    //if(l->reorg) reorg(l->delta, l->w*l->h, size*l->n, l->batch, 0);
    }
}

void detection_layer::backward(network net){
  detection_layer* l = this;
  axpy_cpu(l->batch*l->inputs, 1, l->delta, 1, net.delta, 1);
}

void get_detection_detections(layer* l, int w, int h, float thresh, detection *dets){
  int i,j,n;
  float *predictions = l->output;
  //int per_cell = 5*num+classes;
  for (i = 0; i < l->side*l->side; ++i){
    int row = i / l->side;
    int col = i % l->side;
    for(n = 0; n < l->n; ++n){
      int index = i*l->n + n;
      int p_index = l->side*l->side*l->classes + i*l->n + n;
      float scale = predictions[p_index];
      int box_index = l->side*l->side*(l->classes + l->n) + (i*l->n + n)*4;
      box b;
      b.x = (predictions[box_index + 0] + col) / l->side * w;
      b.y = (predictions[box_index + 1] + row) / l->side * h;
      b.w = pow(predictions[box_index + 2], (l->sqrt?2:1)) * w;
      b.h = pow(predictions[box_index + 3], (l->sqrt?2:1)) * h;
      dets[index].bbox = b;
      dets[index].objectness = scale;
      for(j = 0; j < l->classes; ++j){
        int class_index = i*l->classes;
        float prob = scale*predictions[class_index+j];
        dets[index].prob[j] = (prob > thresh) ? prob : 0;
      }
    }
  }
}
