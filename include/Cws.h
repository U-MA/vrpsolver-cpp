#ifndef VRPSOLVER_CPP_CWS_H
#define VRPSOLVER_CPP_CWS_H

extern "C"
{
#include "vrp_types.h"
}

class Savings
{
public:
    enum
    {
        UNKNOWN = -1
    };
private:
    int value;
    int first;
    int second;

public:
    Savings(void);
    ~Savings(void);

    int getValue(void);
    int getFirst(void);
    int getSecond(void);
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
