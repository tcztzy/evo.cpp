cmake_minimum_required(VERSION 3.25)
project(evo_install_consumer LANGUAGES C)

find_package(evo 0.1 CONFIG REQUIRED)
add_executable(evo-install-consumer main.c)
target_link_libraries(evo-install-consumer PRIVATE evo::evo)
