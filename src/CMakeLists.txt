    add_library(dn
    im2col.cpp

    col2im.cpp

    activations.cpp
    gemm.cpp
    yolo_layer.cpp

    list.cc
#    local_layer.cpp
    shortcut_layer.cpp
    l2norm_layer.cpp
    logistic_layer.cpp
    activation_layer.cpp
    upsample_layer.cpp
    deconvolutional_layer.cpp

    convolutional_layer.cpp
    normalization_layer.cpp
    cuda.cpp
    weights.cpp
    cost_layer.cpp

    crnn_layer.cpp
    rnn_layer.cpp
    gru_layer.cpp
    lstm_layer.cpp
    connected_layer.cpp
    softmax_layer.cpp
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
