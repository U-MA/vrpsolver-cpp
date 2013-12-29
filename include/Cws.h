#ifndef VRPSOLVER_CPP_CWS_H
#define VRPSOLVER_CPP_CWS_H

extern "C"
{
#include "vrp_types.h"
}

typedef std::pair<int, int> EDGE;

class Savings
{
public:
    enum
    {
        UNKNOWN = -1
    };
private:
    int value;
    EDGE edge;

public:
    Savings(void);
    ~Savings(void);

    int getValue(void);
    EDGE getEdge(void);
};

class SavingsList
{
private:
    int size;

public:
    SavingsList(vrp_problem *vrp);
    ~SavingsList(void);

    int getSize(void);
};

#endif /* VRPSOLVER_CPP_CWS_H */
