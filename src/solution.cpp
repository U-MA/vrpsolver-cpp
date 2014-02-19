#include <iostream>

#include "solution.h"

void Solution::Copy(Solution& solution_copy) const
{
    for (int i=0; i < kMaxVehicleSize; i++)
        solution_copy.vehicles_[i] = vehicles_[i];

    solution_copy.current_vehicle_id_ = current_vehicle_id_;
    solution_copy.customer_size_      = customer_size_;
    solution_copy.vehicle_size_       = vehicle_size_;
}

Vehicle *Solution::CurrentVehicle()
{
    return &vehicles_[current_vehicle_id_];
}

void Solution::ChangeVehicle()
{
    ++current_vehicle_id_;
}

bool Solution::IsFeasible() const
{
    return IsFinish();
}

bool Solution::IsFinish() const
{
    /* 用意されている車両を使いきった */
    if (current_vehicle_id_ >= vehicle_size_)
        return true;

    /* 全ての顧客を訪問したかの確認 */
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

bool Solution::IsVisit(int customer_id) const
{
    for (int i=0; i <= current_vehicle_id_; i++)
        if (vehicles_[i].is_visit(customer_id))
            return true;
    return false;
}

int Solution::ComputeTotalCost(const BaseVrp& vrp) const
{
    int total_cost = 0;
    for (int i=0; i <= current_vehicle_id_; i++)
        total_cost += vehicles_[i].computeCost(vrp);
    return total_cost;
}

