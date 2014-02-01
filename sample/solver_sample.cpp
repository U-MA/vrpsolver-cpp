#include "solver.h"

int main(int argc, char **argv)
{
    Solver solver;

    solver.setProblem("../Vrp-All/E/E-n13-k4.vrp");
    solver.setSeed(2013);
    solver.setMctsIterationCount(1000);
    solver.setSimulationCount(1000);
    solver.run();

    return 0;
}
