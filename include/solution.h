#ifndef VRPSOLVER_CPP_SOLUTION_H
#define VRPSOLVER_CPP_SOLUTION_H

#include "base_vrp.h"
#include "vehicle.h"

class Solution
{
public:
    Solution(const BaseVrp& vrp) : vehicles_(),
                                   current_vehicle_(&vehicles_[0]),
                                   current_vehicle_id_(0),
                                   customer_size_(vrp.customer_size()),
                                   vehicle_size_(vrp.vehicle_size()) {}

    void Copy(Solution& solution_copy) const;

    Vehicle *current_vehicle() const { return current_vehicle_; }

    /* 現在走行しているvehicleを変更 */
    void ChangeVehicle();

    /* solutionが適切なものであればtrue
     * 例えば全ての顧客を訪問していなければ適切なsolutionでないので
     * falseを返す */
    bool IsFeasible() const;

    /* 終了状態であればtrue */
    bool IsFinish() const;

    /* customer_idを訪問していればtrue */
    bool IsVisit(int customer_id) const;


    /* 全てのvehicleのコストを返す */
    int ComputeTotalCost(const BaseVrp& vrp) const;

private:
    static const int kMaxVehicleSize = 20;

    Vehicle vehicles_[kMaxVehicleSize];
    Vehicle *current_vehicle_;
    int current_vehicle_id_;
    int customer_size_;
    int vehicle_size_;
};

#endif /* VRPSOLVER_CPP_SOLUTION_H */
