#include "Cws.h"

extern "C"
{
#include "vrp_macros.h"
}

Savings::Savings(void)
{
    value = 0;
    edge  = EDGE(UNKNOWN, UNKNOWN);
}

Savings::~Savings(void)
{
}

static int computeValue(vrp_problem *vrp, int first, int second)
{
    return vrp->dist.cost[INDEX(0, first)] + vrp->dist.cost[INDEX(0, second)] -
           vrp->dist.cost[INDEX(first, second)];
}

void Savings::set(vrp_problem *vrp, int first, int second)
{
    value = computeValue(vrp, first, second);
    edge  = EDGE(first, second);
}

int Savings::getValue(void)
{
    return value;
}

EDGE Savings::getEdge(void)
{
    return edge;
}

bool Savings::operator<(const Savings& s)
{
    return value < s.value;
}

bool Savings::operator>(const Savings& s)
{
    return value > s.value;
}

bool Savings::operator<=(const Savings& s)
{
    return value <= s.value;
}

bool Savings::operator>=(const Savings& s)
{
    return value >= s.value;
}

/***************************************************/
/***************************************************/


SavingsList::SavingsList(vrp_problem *vrp)
{
    if (vrp == NULL)
    {
        size = 0;
        return;
    }

    size = vrp->edgenum;
    savings.set(vrp, 1, 2);
}

SavingsList::~SavingsList(void)
{
}

int SavingsList::getSize(void)
{
    return size;
}

EDGE SavingsList::getEdge(void)
{
    return savings.getEdge();
}
