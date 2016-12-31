cmake_minimum_required(VERSION 3.2.1)
project(MyOpt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

if(NOT EIGEN_INC_DIR)
    include_directories(~/.softwares/linear_algebra/eigen3.3/include/eigen3/)
else()
    include_directories(${EIGEN_INC_DIR})
endif()

set(SRC test.cpp
        MyOpt.cpp
        def.cpp)
set(EXE testopt)
add_executable(${EXE} ${SRC})
set_property(TARGET ${EXE} PROPERTY CXX_STANDARD 11)

# debug macro
option(MYDEBUG "Debug macro for optimizer" OFF)
if(MYDEBUG)
    message(STATUS "Open debug marcro")
    add_definitions(-DMYDEBUG)
    if(DEFINED RAND_SEED)
        message(STATUS "Rand seed is ${RAND_SEED}")
        add_definitions(-DDEBUG_RAND_SEED=${RAND_SEED})
    else()
        add_definitions(-DDEBUG_RAND_SEED=3)
    endif()
else()
    message(STATUS "Close debug marcro")
    remove_definitions(-DMYDEBUG)
    remove_definitions(-DDEBUG_RAND_SEED)
endif()
