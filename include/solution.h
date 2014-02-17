#ifndef VRPSOLVER_CPP_SOLUTION_H
#define VRPSOLVER_CPP_SOLUTION_H

#include "base_vrp.h"
#include "vehicle.h"

class Solution
{
public:
    Solution(const BaseVrp& vrp);
    ~Solution();

    Vehicle *current_vehicle() const { return current_vehicle_; }

    /* solutionが適切なものであればtrue
     * 例えば全ての顧客を訪問していなければ適切なsolutionでないので
     * falseを返す */
    bool IsFeasible() const;

    /* 終了状態であればtrue */
    bool IsFinish() const;

    /* customer_idを訪問していればtrue */
    bool IsVisit(int customer_id) const;

    /* 現在走行しているvehicleを変更 */
    void ChangeVehicle();

    /* 全てのvehicleのコストを返す */
    int ComputeTotalCost() const;

private:
    Vehicle vehicles_[20];
    Vehicle *current_vehicle_;
    int current_vehicle_id_;
    int max_vehicle_size_;
};

#endif /* VRPSOLVER_CPP_SOLUTION_H */
