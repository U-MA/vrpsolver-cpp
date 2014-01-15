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

    vrp_problem *vrp_;
    long        seed_;
    int         count_;
    int         simulation_count_;
};

#endif /* VRPSOLVER_CPP_SOLVER_H */
