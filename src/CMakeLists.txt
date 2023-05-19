add_definitions(-DGPU)

add_library(dn
    im2col.cpp

    activation_kernels.cu
    avgpool_layer_kernels.cu
    blas_kernels.cu
    #crop_layer_kernels.cu
    #deconvolutional_kernels.cu
    #convolutional_kernels.cu


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
      #      set_target_properties(tgt PROPERTIES CUDA_ARCHITECTURES "35;50;72")
      #target_compile_options(dn -DGPU)
      target_link_libraries(dn cuda cudart cublas curand)
