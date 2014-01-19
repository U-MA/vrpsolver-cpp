#include <string.h>
#include <stdio.h>

#include "vehicle_manager.h" 

VehicleManager VehicleManager::copy(void) const
{
    VehicleManager vm_copy;

    vm_copy.size_ = size_;

    /* 使ってる分のVehicleだけコピー */
    for (int i=0; i < size_; i++)
        vm_copy.vehicle_[i] = vehicle_[i].copy();

    const size_t is_visit_bytes = kCustomerMax * sizeof(bool);
    memcpy(vm_copy.is_visit_, is_visit_, is_visit_bytes);

    return vm_copy;
}

int VehicleManager::size(void) const
{
    return size_;
}

bool VehicleManager::isVisit(int customer) const
{
    return is_visit_[customer-1];
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    const int customer_size = vrp->vertnum-1;
    for (int i=1; i <= customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
}

bool VehicleManager::nextVehicleRemain(const vrp_problem *vrp) const
{
    return (size_ < vrp->numroutes);
}

bool VehicleManager::isFinish(const vrp_problem *vrp) const
{
    /* 訪問可能な顧客が存在 */
    for (int i=1; i < vrp->vertnum; i++)
        if (!isVisit(i) && canVisit(vrp, i))
            return false;

    if (nextVehicleRemain(vrp)) return false;

    return true;
}

/* [命名変更希望]
 * capacity制限についての確認なので、その旨を伝えるような
 * 関数名が良い
 * constraint, restriction, limitation
 * とか使えるかも */
bool VehicleManager::canVisit(const vrp_problem *vrp, int customer) const
{
    return (vehicle_[size_-1].capacity() + vrp->demand[customer] <= vrp->capacity);
}


bool VehicleManager::changeVehicle(const vrp_problem *vrp)
{
    if (!nextVehicleRemain(vrp)) return false;

    size_++;
    return true;
}

bool VehicleManager::move(const vrp_problem *vrp, int move)
{
    /* 抽象度が一致してない
     * 条件を関数化する方がよいと思う */
    if (move == kChange)
        return changeVehicle(vrp);

    if (isVisit(move))        return false;
    if (!canVisit(vrp, move)) return false;

    vehicle_[size_-1].visit(vrp, move);
    return (is_visit_[move-1] = true);
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int total_cost = 0;
    for (int i=0; i < size_; i++)
        total_cost += vehicle_[i].computeCost(vrp);

    return total_cost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle_[i].print();
    }
}