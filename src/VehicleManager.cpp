#include <string.h>
#include <stdio.h>

#include "VehicleManager.h" 

VehicleManager::VehicleManager(void)
{
    size_ = 1;
    for (int i=0; i < kCustomerMax; i++)
        is_visit_[i] = false;
}

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

/* [命名変更希望]
 * capacity制限についての確認なので、その旨を伝えるような
 * 関数名が良い */
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

    /* 訪問済 */
    if (is_visit_[move-1] == true) return false;

    /* capacity制限を超過 */
    if ((vehicle_[size_-1].capacity() + vrp->demand[move]) > vrp->capacity) return false;

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
