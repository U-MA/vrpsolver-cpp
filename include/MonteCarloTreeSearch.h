#ifndef VRPSOLVER_CPP_MONTECARLOTREESEARCH_H
#define VRPSOLVER_CPP_MONTECARLOTREESEARCH_H

extern "C"
{
#include "vrp_types.h"
}

#include "Node.h"
#include "VehicleManager.h"

class MonteCarloTree
{
public:
    MonteCarloTree(void);
    ~MonteCarloTree(void);

    void init(void);
    void search(const vrp_problem *vrp, const VehicleManager& vm, const Vehicle& v);
    int  next(void);

private:
    int size_;
    Node *node;
};

#endif /* VRPSOLVER_CPP_MONTECARLOTREESEARCH_H */
