#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

class Node
{
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
    void expand(const vrp_problem *vrp, VehicleManager& vm);
    void update(int value);

    /* privateにしたいがテストのためpublicにしている */
    Node *select(void);

    /* for MonteCarloTreeSearch */
    //void search(const vrp_problem *vrp, const VehicleManager& vm, const Vehicle& v);
    void search(const vrp_problem *vrp, const VehicleManager& vm);
    int  next(void) const;

private:
    double computeUcb(int parentCount);

    int  customer_;
    int  count_;
    int  childSize_;
    int  value_;
    Node *child;
};

#endif /* VRPSOLVER_CPP_NODE_H */
