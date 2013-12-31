#ifndef VRPSOLVER_CPP_CWS_H
#define VRPSOLVER_CPP_CWS_H

#include <queue>

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
    Savings(const Savings& savings);
    ~Savings(void);

    void set(vrp_problem *vrp, int first, int second);
    int getValue(void);
    EDGE getEdge(void) const;

    bool operator<(const Savings& s) const;
    bool operator>(const Savings& s) const;
    bool operator<=(const Savings& s) const;
    bool operator>=(const Savings& s) const;
};

class SavingsList
{
private:
    std::priority_queue<Savings> savings;

public:
    SavingsList(vrp_problem *vrp);
    ~SavingsList(void);

    int getSize(void);
    EDGE getEdge(void) const;
};

#endif /* VRPSOLVER_CPP_CWS_H */
