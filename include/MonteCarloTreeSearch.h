#ifndef VRPSOLVER_CPP_MONTECARLOTREESEARCH_H
#define VRPSOLVER_CPP_MONTECARLOTREESEARCH_H

extern "C"
{
#include "vrp_types.h"
}

#include "VehicleManager.h"

namespace MCTS
{
    void MonteCarloTreeSearch(const vrp_problem *vrp, const VehicleManager& vm);
}

#endif /* VRPSOLVER_CPP_MONTECARLOTREESEARCH_H */
