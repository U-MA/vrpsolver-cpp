#ifndef CPPSOLVER_CPP_WRAPPER_VRP_H
#define CPPSOLVER_CPP_WRAPPER_VRP_H

extern "C"
{
#include "vrp_types.h"
}

vrp_problem *createVrpFrom(char *infile);
void destroyVrp(vrp_problem *vrp);

#endif /* CPPSOLVER_CPP_WRAPPER_VRP_H */
