add_compile_options(-Wall -Wextra
    -Wno-pedantic -Wno-write-strings -Wno-unused-variable -Wno-unused-parameter
    -Wno-sign-compare
    )
    
    
    #    add_executable(art art.cpp)
    add_executable(darknet darknet.cpp yolo.cpp super.cpp # lsd.cpp
    detector.cpp cifar.cpp go.cpp rnn.cpp
     coco.cpp )
    target_link_libraries(darknet dn)
    
