#include <iostream>

#include "solution.h"

void Solution::ChangeVehicle()
{
    current_vehicle_ = &vehicles_[++current_vehicle_id_];
}

bool Solution::IsFeasible() const
{
    return IsFinish();
}

bool Solution::IsFinish() const
{
    for (int i=1; i <= customer_size_; i++)
    {
        for (int j=0; j <= current_vehicle_id_; j++)
        {
            if (vehicles_[j].is_visit(i))
                break;
            if (j == current_vehicle_id_)
                return false;
        }
    }
    return true;
}
