#include "Cws.h"

Savings::Savings(void)
{
    value = 0;
    first = second = UNKNOWN;
}

Savings::~Savings(void)
{
}

int Savings::getValue(void)
{
    return value;
}

int Savings::getFirst(void)
{
    return first;
}

int Savings::getSecond(void)
{
    return second;
}

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
