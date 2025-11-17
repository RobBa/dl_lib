#include "hello.h"

//void World::set(std::string msg) { 
//    this->msg = msg; 
//}

/*std::string World::greet() { 
    return msg; 
}*/

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(hello)
{
    /*class_<World>("World")
        .def("greet", &World::greet)
        .def("set", &World::set)
    ;*/
    def("greet", greet);
};