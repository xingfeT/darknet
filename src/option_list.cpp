#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

list *read_data_cfg(const char *filename){
    FILE *file = fopen(filename, "r");
    if(file == 0) {
      file_error(filename);
    }

    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
              if(!read_option(line, options)){
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
              }
              break;
        }
    }

    fclose(file);
    return options;
}

metadata get_metadata(char *file){
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find(options, "names", 0);
    if(!name_list) name_list = option_find(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}
