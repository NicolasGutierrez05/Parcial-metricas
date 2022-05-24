cmake_minimum_required(VERSION 3.5)

project(Proyecto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (project_sources
    main.cpp
    RL/linealregresion.cpp
    extraccion/extraerdata.cpp
    )
set (project_headers
    RL/linealregresion.h
    extraccion/extraerdata.h)


add_executable(${PROJECT_NAME} ${project_sources} ${project_headers})
