cmake_minimum_required(VERSION 3.2.1)
project(MyOpt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()
message(STATUS "Build type for MyOpt is ${CMAKE_BUILD_TYPE}")

if(NOT EIGEN_INC_DIR)
    include_directories(~/.softwares/linear_algebra/eigen3.3/include/eigen3/)
else()
    include_directories(${EIGEN_INC_DIR})
endif()

set(LIBSRC MyOpt.cpp def.cpp)
set(EXESRC test.cpp)
set(EXE testopt)
set(LIB MyOpt)
add_executable(${EXE} ${EXESRC})
add_library(${LIB} ${LIBSRC})
target_link_libraries(${EXE} ${LIB})
set_property(TARGET ${EXE} ${LIB} PROPERTY CXX_STANDARD 11)

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

# install program, libs, headers and docs
if(CMAKE_INSTALL_PREFIX)
    message(STATUS "Cmake install prefix for MyOpt: ${CMAKE_INSTALL_PREFIX}")
    install(TARGETS ${LIB} ${EXE}
        RUNTIME DESTINATION MyOpt/bin
        ARCHIVE DESTINATION MyOpt/lib
        LIBRARY DESTINATION MyOpt/lib)
    install(FILES MyOpt.h def.h
        DESTINATION MyOpt/inc
        PERMISSIONS OWNER_READ GROUP_READ)
    install(FILES README.md
        DESTINATION MyOpt/doc
        PERMISSIONS OWNER_READ GROUP_READ)
endif()
