#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list(){
  list *l = (list *)calloc(1, sizeof(list));
  return l;
}


void *list_pop(list *l){
    if(!l->back) return nullptr;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    return val;
}

void list_insert(list *l, void *val){
  node *n = (node*)malloc(sizeof(node));
  n->val = val;
  n->next = 0;
  
  if(!l->back){
    l->front = n;
    n->prev = 0;
  }else{
    l->back->next = n;
    n->prev = l->back;
  }
  l->back = n;
  ++l->size;
}

void free_node(node *n){
  node *next;
  while(n) {
    next = n->next;
    free(n);
    n = next;
  }
}

void free_list(list *l){
  free_node(l->front);
  free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l){
  void **a = (void**)calloc(l->size, sizeof(void*));
  int count = 0;
  node *n = l->front;
  while(n){
    a[count++] = n->val;
    n = n->next;
  }
  return a;
}
