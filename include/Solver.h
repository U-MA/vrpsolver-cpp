#ifndef VRPSOLVER_CPP_SOLVER_H
#define VRPSOLVER_CPP_SOLVER_H

class Solver
{
public:
    Solver(void) : vrp_(NULL), seed_(2013), count_(1000),
                   simulation_count_(1) {};
    ~Solver(void);

    void setProblem(char *filename);
    void setSeed(long seed);
    void setMctsIterationCount(int count);
    void setSimulationCount(int count);
    void run(void);

private:
    vrp_problem *vrp_;
    long        seed_;
    int         count_;
    int         simulation_count_;
};

#endif /* VRPSOLVER_CPP_SOLVER_H */
