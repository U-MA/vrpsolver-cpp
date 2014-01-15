#ifndef VRPSOLVER_CPP_SOLVER_H
#define VRPSOLVER_CPP_SOLVER_H

class Solver
{
public:
    void setProblem(char *filename);
    void setSeed(long seed);
    void setMctsIterationCount(int count);
    void setSimulationCount(int count);
    void run(void);
    void freeProblem(void);

private:
    void cookMember(void);

    vrp_problem *vrp;
    long        seed;
    int         count;
    int         simulationCount;
};

#endif /* VRPSOLVER_CPP_SOLVER_H */
