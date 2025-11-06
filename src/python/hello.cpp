#include "hello.h"

void World::set(std::string msg) { 
    this->msg = msg; 
}

std::string World::greet() { 
    return msg; 
}