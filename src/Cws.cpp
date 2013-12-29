#include "Cws.h"

Savings::Savings(void)
{
    value = 0;
    edge = EDGE(UNKNOWN, UNKNOWN);
}

Savings::~Savings(void)
{
}

int Savings::getValue(void)
{
    return value;
}

EDGE Savings::getEdge(void)
{
    return edge;
}


/***************************************************/
/***************************************************/


SavingsList::SavingsList(vrp_problem *vrp)
{
    size = 0;
}

SavingsList::~SavingsList(void)
{
}

int SavingsList::getSize(void)
{
    return size;
}
