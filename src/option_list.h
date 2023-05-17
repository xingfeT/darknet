#pragma once

#include "list.h"
#include <string>

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
void option_unused(list *l);
list *read_data_cfg(char *filename);

metadata get_metadata(char *file);


int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

void option_insert(list *l, char *key, char *val){
  kvp *p = (kvp *)malloc(sizeof(kvp));
  p->key = key;
  p->val = val;
  p->used = 0;
  list_insert(l, p);
}

void option_unused(list *l){
  node *n = l->front;
  while(n){
    kvp *p = (kvp *)n->val;
    if(!p->used){
      fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
    }
    n = n->next;
  }
}

char *option_find(list *l, const char *key){
  node *n = l->front;
  while(n){
    kvp *p = (kvp *)n->val;
    if(strcmp(p->key, key) == 0){
      p->used = 1;
      return p->val;
    }
    n = n->next;
  }
  return 0;
}

// template<typename T>
// T option_find_value(list *l, const char * key, T def, bool quiet=false){
//   char *v = option_find(l, key);
//   if(v){
//     std::stringstream convert(std::string(s));
//     T value;
//     convert >> value;
//     return value;
//   }
//   if(!quiet){
//     fprintf(stderr, "%s: Using default '%d'\n", key, def);
//   }
//   return def;
// }

// template<>
// char *option_find_value(list *l, const char *key, const char *def,  bool quiet){
//     char *v = option_find(l, key);
//     if(v) return v;
//     if(!quiet) {
//       fprintf(stderr, "%s: Using default '%s'\n", key, def);
//     }
//     return strdup(def);
// }

template<typename T>
T option_find(list *l, const char * key, T def){
  char *v = option_find(l, key);
  if(v){
    std::stringstream convert(std::string(v));
    T value;
    //todo
    //convert >> value;
    return value;
  }

  fprintf(stderr, "%s: Using default '%d'\n", key, def);
  return def;
}

char *option_find_str(list *l, const char *key, const char *def){
    char *v = option_find(l, key);
    if(v) return v;
    fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return strdup(def);
}

template<typename T>
T option_find_quiet(list *l, const char *key, T const & def){
  return option_find(l, key, def, true);
}
