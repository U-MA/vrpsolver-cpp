#include <stdlib.h>

extern "C"
{
#include "vrp_macros.h"
}

#include "VrpProblems.h"

static void Vrp_SetCost(vrp_problem *vrp, int first, int second, int value)
{
    vrp->dist.cost[INDEX(first, second)] = value;
}

vrp_problem *VrpProblem::AMCT2CVRP(void)
{
    vrp_problem *vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp->dist.cost = (int *)calloc(100, sizeof(int));
    vrp->demand    = (int *)calloc(100, sizeof(int));

    vrp->vertnum   = 6;
    vrp->edgenum   = vrp->vertnum * (vrp->vertnum-1) / 2;
    vrp->numroutes = 2;

    Vrp_SetCost(vrp, 0, 1, 28);
    Vrp_SetCost(vrp, 0, 2, 31);
    Vrp_SetCost(vrp, 0, 3, 20);
    Vrp_SetCost(vrp, 0, 4, 25);
    Vrp_SetCost(vrp, 0, 5, 34);
    Vrp_SetCost(vrp, 1, 2, 21);
    Vrp_SetCost(vrp, 1, 3, 29);
    Vrp_SetCost(vrp, 1, 4, 26);
    Vrp_SetCost(vrp, 1, 5, 20);
    Vrp_SetCost(vrp, 2, 3, 38);
    Vrp_SetCost(vrp, 2, 4, 20);
    Vrp_SetCost(vrp, 2, 5, 32);
    Vrp_SetCost(vrp, 3, 4, 30);
    Vrp_SetCost(vrp, 3, 5, 27);
    Vrp_SetCost(vrp, 4, 5, 25);

    vrp->capacity  = 100;
    vrp->demand[1] = 37;
    vrp->demand[2] = 35;
    vrp->demand[3] = 30;
    vrp->demand[4] = 25;
    vrp->demand[5] = 32;

    return vrp;
}

vrp_problem *VrpProblem::E_n13_k4(void)
{
    vrp_problem *vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    char infile[200];
    sprintf(infile, "Vrp-All/E/E-n13-k4.vrp");
    vrp_io(vrp, infile);
    vrp->numroutes = 4;
    return vrp;
}

vrp_problem *VrpProblem::E_n51_k5(void)
{
    vrp_problem *vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    char infile[200];
    sprintf(infile, "Vrp-All/E/E-n51-k5.vrp");
    vrp_io(vrp, infile);
    vrp->numroutes = 5;
    return vrp;
}

vrp_problem *VrpProblem::E_n101_k14(void)
{
    vrp_problem *vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    char infile[200];
    sprintf(infile, "Vrp-All/E/E-n101-k14.vrp");
    vrp_io(vrp, infile);
    vrp->numroutes = 14;
    return vrp;
}


void VrpProblem::teardown(vrp_problem *vrp)
{
    if (vrp->dist.cost != NULL)
    {
        free(vrp->dist.cost); vrp->dist.cost = NULL;
    }

    if (vrp->dist.coordx != NULL)
    {
        free(vrp->dist.coordx); vrp->dist.coordx = NULL;
    }

    if (vrp->dist.coordy != NULL)
    {
        free(vrp->dist.coordy); vrp->dist.coordy = NULL;
    }

    if (vrp->demand != NULL)
    {
        free(vrp->demand); vrp->demand = NULL;
    }

    if (vrp->posx != NULL)
    {
        free(vrp->posx); vrp->posx = NULL;
    }

    if (vrp->posy != NULL)
    {
        free(vrp->posy); vrp->posy = NULL;
    }

    if (vrp != NULL)
    {
        free(vrp); vrp = NULL;
    }
}
