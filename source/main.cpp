#include "opencl_program.hpp"

int main(int argc, char** argv)
{
    CL_Program p("simple_copy.cl");
    p.loadProgram();
    p.runKernel();
    return 0;
}

