#ifndef VRPSOLVER_CPP_SOLVER_H
#define VRPSOLVER_CPP_SOLVER_H

#include <cstdlib>

#include "host_vrp.h"

class Solver
{
public:
    Solver(void) : vrp_(), seed_(2013), count_(1000),
                   simulation_count_(1) {};

    void setProblem(const char *filename);
    void setSeed(long seed);
    void setMctsIterationCount(int count);
    void setSimulationCount(int count);
    void run(void);

private:
    void printRunParameter(void);

    HostVrp     vrp_;
    long        seed_;
    int         count_;
    int         simulation_count_;
};

#endif /* VRPSOLVER_CPP_SOLVER_H */
