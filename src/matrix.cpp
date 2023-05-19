#include "matrix.h"
#include "utils.h"
#include "blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

matrix::~matrix(){
  free(vals);
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k){
  int *indexes = (int*)calloc(k, sizeof(int));
  int n = truth.cols;
  int i,j;
  int correct = 0;
  for(i = 0; i < truth.rows; ++i){
    top_k(guess(i), n, k, indexes);
    for(j = 0; j < k; ++j){
      int class1 = indexes[j];
      if(truth(i, class1)){
        ++correct;
        break;
      }
    }
  }
  free(indexes);
  return (float)correct/truth.rows;
}

void matrix::scale(float scale){
  for(int i = 0; i < rows * cols; ++i){
    vals[i] *= scale;
  }
}

void matrix::resize(int size){
  int i;
  if (rows == size) return ;
  vals = (float*)realloc(vals, size*sizeof(float)*cols);
  rows = size;
  return ;
}

// +=
matrix& matrix::operator+=(matrix const &from){
  assert(from.rows == rows && from.cols == cols);
  for(int i = 0; i < from.rows * from.cols; ++i){
    vals[i] += from.vals[i];
  }
  return *this;
}

matrix::matrix(matrix const &m){
  rows = m.rows;
  cols = m.cols;
  vals = (float*)calloc(m.rows*m.cols, sizeof(float));
  memmove(m.vals, vals, sizeof(float) * m.rows*m.cols);
}

matrix::matrix(int r, int c):rows(r), cols(c){
  vals = (float*)calloc(rows*cols, sizeof(float));
}

matrix hold_out_matrix(matrix *m, int n){
  int i;
  matrix h(n, m->cols);
  for(i = 0; i < n; ++i){
    int index = rand()%m->rows;
    //h.vals[i] = m->vals[index];
    //m->vals[index] = m->vals[--(m->rows)];
  }
  return h;
}

float *pop_column(matrix *m, int c){
  float *col = nullptr;//calloc(m->rows, sizeof(float));
    int i, j;
    for(i = 0; i < m->rows; ++i){
      //col[i] = m->vals[i][c];
        for(j = c; j < m->cols-1; ++j){
          //m->vals[i][j] = m->vals[i][j+1];
        }
    }
    --m->cols;
    return col;
}

matrix matrix::from_csv(const char *filename){
  FILE *fp = fopen(filename, "r");
  if(!fp) file_error(filename);

  matrix m;
  m.cols = -1;

  char *line;

  int n = 0;
  int size = 1024;
  m.vals = (float*)calloc(size, sizeof(float*));
  while((line = fgetl(fp))){
    if(m.cols == -1) m.cols = count_fields(line);
    if(n == size){
      size *= 2;
      m.vals = (float*)realloc(m.vals, size*sizeof(float*));
    }
    //m.vals[n] = parse_fields(line, m.cols);
    free(line);
    ++n;
  }
  m.vals = (float*)realloc(m.vals, n*sizeof(float*));
  m.rows = n;
  return m;
}

void matrix::to_csv(){
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      if(j > 0) printf(",");
      //printf("%.17g", vals[i][j]);
    }
    printf("\n");
  }
}

void matrix::print() const {
  int i, j;
  printf("%d X %d Matrix:\n",rows, cols);
  printf(" __");
  for(j = 0; j < 16*cols-1; ++j) printf(" ");
  printf("__ \n");

  printf("|  ");
  for(j = 0; j < 16*cols-1; ++j) printf(" ");
  printf("  |\n");

  for(i = 0; i < rows; ++i){
    printf("|  ");
    for(j = 0; j < cols; ++j){
      printf("%15.7f ", (*this)(i,j));
    }
    printf(" |\n");
  }
  printf("|__");
  for(j = 0; j < 16*cols-1; ++j) printf(" ");
  printf("__|\n");
}
