#ifndef VRPSOLVER_CPP_TEST_VRPPROBLEM_H
#define VRPSOLVER_CPP_TEST_VRPPROBLEM_H

extern "C"
{
#include "vrp_io.h"
#include "vrp_types.h"
}


namespace VrpProblem
{
    /* Applying Monte Carlo Techniques to Capacitated Vehicle
     * Routing Problem p.24の例の問題 */
    vrp_problem *AMCT2CVRP(void);

    /* From Vrp-All/E/ */
    vrp_problem *E_n13_k4(void);
    vrp_problem *E_n51_k5(void);
    vrp_problem *E_n101_k14(void);

    void teardown(vrp_problem *vrp);
}

#endif /* VRPSOLVER_CPP_TEST_VRPPROBLEM_H */
