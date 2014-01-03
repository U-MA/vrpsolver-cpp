#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

class Node
{
private:
    double computeUcb(void);

    int  customer_;
    int  count_;
    int  childSize_;
    int  value_;
    Node *child;

public:
    Node(void);
    ~Node(void);

    /* getter */
    int customer(void) const;
    int count(void) const;
    int childSize(void) const;
    int value(void) const;

    bool isLeaf(void) const;
    void expand(int childSize);
    Node *select(void);
    void update(int value);

    /* for MonteCarloTreeSearch */
    void init(void);
    void search(const vrp_problem *vrp, const VehicleManager& vm, const Vehicle& v);
    int  next(void) const;
};

#endif /* VRPSOLVER_CPP_NODE_H */
