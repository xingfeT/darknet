#pragma once
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
struct route_layer;
