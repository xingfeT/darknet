#include "lstm_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include <initializer_list>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lstm_layer::increment_layer(int steps){
  lstm_layer * l = this;
  
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;
    
}

lstm_layer* make_lstm_layer(int batch, int inputs, int outputs,
                            int steps, int batch_normalize, int adam){
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    lstm_layer* l = new lstm_layer();
    
    l->batch = batch;
    l->type = LSTM;
    l->steps = steps;
    l->inputs = inputs;
    for(layer** u : {&(l->uf),  &(l->ui), &(l->ug), &(l->uo), &(l->wf), &(l->wi), &(l->wg), &(l->wo)}){
      *u = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize, adam);
      (*u)->batch = batch;
    }

    l->batch_normalize = batch_normalize;
    l->outputs = outputs;

    l->output = (float*)calloc(outputs*batch*steps, sizeof(float));
    l->state = (float*)calloc(outputs*batch, sizeof(float));

    l->prev_state_cpu =  (float*)calloc(batch*outputs, sizeof(float));
    l->prev_cell_cpu =   (float*)calloc(batch*outputs, sizeof(float));
    l->cell_cpu =        (float*)calloc(batch*outputs*steps, sizeof(float));

    l->f_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l->i_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l->g_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l->o_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l->c_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    l->h_cpu =           (float*)calloc(batch*outputs, sizeof(float));
    
    l->temp_cpu =        (float*)calloc(batch*outputs, sizeof(float));
    l->temp2_cpu =       (float*)calloc(batch*outputs, sizeof(float));
    l->temp3_cpu =       (float*)calloc(batch*outputs, sizeof(float));
    l->dc_cpu =          (float*)calloc(batch*outputs, sizeof(float));
    l->dh_cpu =          (float*)calloc(batch*outputs, sizeof(float));


    return l;
}

void lstm_layer::update(update_args a){
  lstm_layer* l = this;
  
  for(auto& u : {l->uf,  l->ui, l->ug, l->uo, l->wf, l->wi, l->wg, l->wo}){
    u->update(a);
  }
}

void lstm_layer::forward(network state){
  lstm_layer* l = this;
  
    network s = { 0 };
    s.train = state.train;
    int i;
    for(auto& u : {l->uf,  l->ui, l->ug, l->uo, l->wf, l->wi, l->wg, l->wo}){
      fill_cpu(l->outputs * l->batch * l->steps, 0, u->delta, 1);
    }

    
    if (state.train) {
      fill_cpu(l->outputs * l->batch * l->steps, 0, l->delta, 1);
    }

    for (i = 0; i < l->steps; ++i) {
        s.input = l->h_cpu;
        for(auto& u : {l->wf, l->wi, l->wg, l->wo}){
          u->forward(s);
        }

        s.input = state.input;
        for(auto& u : {l->uf, l->ui, l->ug, l->uo}){
          u->forward(s);
        }
        
        copy_cpu(l->outputs*l->batch, wf->output, 1, l->f_cpu, 1);
        axpy_cpu(l->outputs*l->batch, 1, uf->output, 1, l->f_cpu, 1);

        copy_cpu(l->outputs*l->batch, wi->output, 1, l->i_cpu, 1);	
        axpy_cpu(l->outputs*l->batch, 1, ui->output, 1, l->i_cpu, 1);	

        copy_cpu(l->outputs*l->batch, wg->output, 1, l->g_cpu, 1);	
        axpy_cpu(l->outputs*l->batch, 1, ug->output, 1, l->g_cpu, 1);	

        copy_cpu(l->outputs*l->batch, wo->output, 1, l->o_cpu, 1);	
        axpy_cpu(l->outputs*l->batch, 1, uo->output, 1, l->o_cpu, 1);	

        activate_array(l->f_cpu, l->outputs*l->batch, LOGISTIC);		
        activate_array(l->i_cpu, l->outputs*l->batch, LOGISTIC);		
        activate_array(l->g_cpu, l->outputs*l->batch, TANH);			
        activate_array(l->o_cpu, l->outputs*l->batch, LOGISTIC);		

        copy_cpu(l->outputs*l->batch, l->i_cpu, 1, l->temp_cpu, 1);		
        mul_cpu(l->outputs*l->batch, l->g_cpu, 1, l->temp_cpu, 1);		
        mul_cpu(l->outputs*l->batch, l->f_cpu, 1, l->c_cpu, 1);			
        axpy_cpu(l->outputs*l->batch, 1, l->temp_cpu, 1, l->c_cpu, 1);	

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->h_cpu, 1);			
        activate_array(l->h_cpu, l->outputs*l->batch, TANH);		
        mul_cpu(l->outputs*l->batch, l->o_cpu, 1, l->h_cpu, 1);	

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->cell_cpu, 1);		
        copy_cpu(l->outputs*l->batch, l->h_cpu, 1, l->output, 1);

        state.input += l->inputs*l->batch;
        l->output    += l->outputs*l->batch;
        l->cell_cpu      += l->outputs*l->batch;

        for(auto& u : {l->uf,  l->ui, l->ug, l->uo, l->wf, l->wi, l->wg, l->wo}){
          u->increment_layer(1);
        }
    }
}

void lstm_layer::backward(network state){
  lstm_layer* l = this;
  
    network s = { 0 };
    s.train = state.train;
    int i;

    for(auto& u : {l->uf,  l->ui, l->ug, l->uo, l->wf, l->wi, l->wg, l->wo}){
      u->increment_layer(l->steps - 1);
    }
    
    state.input += l->inputs*l->batch*(l->steps - 1);
    if (state.delta) state.delta += l->inputs*l->batch*(l->steps - 1);

    l->output += l->outputs*l->batch*(l->steps - 1);
    l->cell_cpu += l->outputs*l->batch*(l->steps - 1);
    l->delta += l->outputs*l->batch*(l->steps - 1);

    for (i = l->steps - 1; i >= 0; --i) {
      if (i != 0) copy_cpu(l->outputs*l->batch, l->cell_cpu - l->outputs*l->batch, 1, l->prev_cell_cpu, 1);
      copy_cpu(l->outputs*l->batch, l->cell_cpu, 1, l->c_cpu, 1);
      if (i != 0) copy_cpu(l->outputs*l->batch, l->output - l->outputs*l->batch, 1, l->prev_state_cpu, 1);
      copy_cpu(l->outputs*l->batch, l->output, 1, l->h_cpu, 1);

      l->dh_cpu = (i == 0) ? 0 : l->delta - l->outputs*l->batch;

        copy_cpu(l->outputs*l->batch, wf->output, 1, l->f_cpu, 1);			
        axpy_cpu(l->outputs*l->batch, 1, uf->output, 1, l->f_cpu, 1);			

        copy_cpu(l->outputs*l->batch, wi->output, 1, l->i_cpu, 1);			
        axpy_cpu(l->outputs*l->batch, 1, ui->output, 1, l->i_cpu, 1);			

        copy_cpu(l->outputs*l->batch, wg->output, 1, l->g_cpu, 1);			
        axpy_cpu(l->outputs*l->batch, 1, ug->output, 1, l->g_cpu, 1);			

        copy_cpu(l->outputs*l->batch, wo->output, 1, l->o_cpu, 1);			
        axpy_cpu(l->outputs*l->batch, 1, uo->output, 1, l->o_cpu, 1);			

        activate_array(l->f_cpu, l->outputs*l->batch, LOGISTIC);			
        activate_array(l->i_cpu, l->outputs*l->batch, LOGISTIC);		
        activate_array(l->g_cpu, l->outputs*l->batch, TANH);			
        activate_array(l->o_cpu, l->outputs*l->batch, LOGISTIC);		

        copy_cpu(l->outputs*l->batch, l->delta, 1, l->temp3_cpu, 1);		

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);			
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);			

        copy_cpu(l->outputs*l->batch, l->temp3_cpu, 1, l->temp2_cpu, 1);		
        mul_cpu(l->outputs*l->batch, l->o_cpu, 1, l->temp2_cpu, 1);			

        gradient_array(l->temp_cpu, l->outputs*l->batch, TANH, l->temp2_cpu);
        axpy_cpu(l->outputs*l->batch, 1, l->dc_cpu, 1, l->temp2_cpu, 1);		

        copy_cpu(l->outputs*l->batch, l->c_cpu, 1, l->temp_cpu, 1);			
        activate_array(l->temp_cpu, l->outputs*l->batch, TANH);			
        mul_cpu(l->outputs*l->batch, l->temp3_cpu, 1, l->temp_cpu, 1);		
        gradient_array(l->o_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, wo->delta, 1);
        s.input = l->prev_state_cpu;
        s.delta = l->dh_cpu;															
        wo->backward(s);	

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, uo->delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        uo->backward(s);									

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);			
        mul_cpu(l->outputs*l->batch, l->i_cpu, 1, l->temp_cpu, 1);				
        gradient_array(l->g_cpu, l->outputs*l->batch, TANH, l->temp_cpu);		
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, wg->delta, 1);
        
        s.input = l->prev_state_cpu;
        s.delta = l->dh_cpu;														
        wg->backward(s);	

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, ug->delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        ug->backward(s);																

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);			
        mul_cpu(l->outputs*l->batch, l->g_cpu, 1, l->temp_cpu, 1);				
        gradient_array(l->i_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);	
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, wi->delta, 1);
        s.input = l->prev_state_cpu;
        s.delta = l->dh_cpu;
        wi->backward(s);						

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, ui->delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        ui->backward(s);									

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);		
        mul_cpu(l->outputs*l->batch, l->prev_cell_cpu, 1, l->temp_cpu, 1);
        gradient_array(l->f_cpu, l->outputs*l->batch, LOGISTIC, l->temp_cpu);
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, wf->delta, 1);
        s.input = l->prev_state_cpu;
        s.delta = l->dh_cpu;
        wf->backward(s);						

        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, uf->delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        uf->backward(s);									

        copy_cpu(l->outputs*l->batch, l->temp2_cpu, 1, l->temp_cpu, 1);			
        mul_cpu(l->outputs*l->batch, l->f_cpu, 1, l->temp_cpu, 1);				
        copy_cpu(l->outputs*l->batch, l->temp_cpu, 1, l->dc_cpu, 1);				

        state.input -= l->inputs*l->batch;
        if (state.delta) state.delta -= l->inputs*l->batch;
        l->output -= l->outputs*l->batch;
        l->cell_cpu -= l->outputs*l->batch;
        l->delta -= l->outputs*l->batch;

        for(auto& u : {l->uf,  l->ui, l->ug, l->uo, l->wf, l->wi, l->wg, l->wo}){
          u->increment_layer(-1);
        }

    }
}

