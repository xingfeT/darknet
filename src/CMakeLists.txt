    add_library(dn
    im2col.cpp

    activations.cpp
    gemm.cpp


    list.cc
#    local_layer.cpp
#    deconvolutional_layer.cpp
    convolutional_layer.cpp
    normalization_layer.cpp
    cuda.cpp
    weights.cpp

    crnn_layer.cpp
    rnn_layer.cpp
    gru_layer.cpp
    lstm_layer.cpp
    connected_layer.cpp
    softmax_layer.cpp
    yolo_layer.cpp
    iseg_layer.cpp
    region_layer.cpp
    detection_layer.cpp
    crop_layer.cpp
    reorg_layer.cpp
    maxpool_layer.cpp
    avgpool_layer.cpp
    dropout_layer.cpp
    batchnorm_layer.cpp

        tree.cpp parser.cpp box.cpp data.cpp network.cpp layer.cpp image.cpp blas.cpp matrix.cpp utils.cpp)
