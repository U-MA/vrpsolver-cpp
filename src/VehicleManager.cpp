#include <string.h>
#include <stdio.h>

#include "VehicleManager.h" 

VehicleManager::VehicleManager(void)
{
    size_ = 1;
    for (int i=0; i < kCustomerMax; i++)
        isVisit_[i] = false;
}

VehicleManager VehicleManager::copy(void) const
{
    VehicleManager vm_copy;

    vm_copy.size_ = size_;

    /* 使ってる分のVehicleだけコピー */
    for (int i=0; i < size_; i++)
        vm_copy.vehicle_[i] = vehicle_[i].copy();

    const size_t isVisit_bytes = kCustomerMax * sizeof(bool);
    memcpy(vm_copy.isVisit_, isVisit_, isVisit_bytes);

    return vm_copy;
}

int VehicleManager::size(void) const
{
    return size_;
}

bool VehicleManager::isVisit(int customer) const
{
    return isVisit_[customer-1];
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    const int customer_size = vrp->vertnum;
    for (int i=1; i < customer_size; i++)
        if (!isVisit(i)) return false;

    return true;
}

bool VehicleManager::isFinish(const vrp_problem *vrp)
{
    for (int i=1; i < vrp->vertnum; i++)
        if (!isVisit(i) && canVisit(vrp, i))
            return false;

    /* 次の車体があればfalse */
    if (size_ < vrp->numroutes)
        return false;

    return true;
}

bool VehicleManager::canVisit(const vrp_problem *vrp, int customer)
{
    return (vehicle_[size_-1].capacity() + vrp->demand[customer] <= vrp->capacity);
}

bool VehicleManager::move(const vrp_problem *vrp, int move)
{
    /* 車体の変更 */
    if (move == kChange)
    {
        if (size_ == vrp->numroutes)
        {
            /* 次の車体が無い */
            return false;
        }
        else
        {
            size_++;
            return true;
        }
    }

    /* 同じ顧客を再び訪問済 */
    if (isVisit_[move-1] == true) return false;

    /* capacity制限を超過 */
    if ((vehicle_[size_-1].capacity() + vrp->demand[move]) > vrp->capacity) return false;

    vehicle_[size_-1].visit(vrp, move);
    return (isVisit_[move-1] = true);
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int totalCost = 0;
    for (int i=0; i < size_; i++)
        totalCost += vehicle_[i].computeCost(vrp);

    return totalCost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle_[i].print();
    }
}
