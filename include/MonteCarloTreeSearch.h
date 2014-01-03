#ifndef VRPSOLVER_CPP_MONTECARLOTREESEARCH_H
#define VRPSOLVER_CPP_MONTECARLOTREESEARCH_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

namespace MCTS
{
    int MonteCarloTreeSearch(const vrp_problem *vrp, const VehicleManager& vm, int loopCount);
}

#endif /* VRPSOLVER_CPP_MONTECARLOTREESEARCH_H */
