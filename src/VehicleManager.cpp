#include "VehicleManager.h" 

VehicleManager::VehicleManager(void)
{
    size_ = 0;
    ranSize = 1;
}

VehicleManager::~VehicleManager(void)
{
}

int VehicleManager::size(void) const
{
    return size_;
}

void VehicleManager::add(Vehicle& v)
{
    vehicle[size_] = v;
    size_++;
}

VehicleManager VehicleManager::copy(void) const
{
    VehicleManager vm_copy;
    vm_copy.size_ = size_;

    for (int i=0; i < VEHICLE_MAX; i++)
        vm_copy.vehicle[i] = vehicle[i];

    return vm_copy;
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    int customerSize = vrp->vertnum-1;
    for (int i=1; i < customerSize; i++)
        if (!isVisit(i)) return false;

    return true;
}

bool VehicleManager::isVisit(int customer) const
{
    for (int i=0; i < size_; i++)
    {
        if (vehicle[i].isVisit(customer))
            return true;
    }
    return false;
}

bool VehicleManager::move(const vrp_problem *vrp, int move)
{
    /* 車体の変更 */
    if (move == CHANGE)
    {
        if (ranSize == vrp->numroutes)
        {
            /* 次の車体が無い */
            return false;
        }
        else
        {
            ranSize++;
            return true;
        }
    }

    /* 同じ顧客を再び訪問済 */
    if (isVisit_[move] == true) return false;

    /* capacity制限を超過 */
    if ((vehicle[ranSize-1].quantity() + vrp->demand[move]) > vrp->capacity) return false;

    vehicle[ranSize-1].visit(vrp, move);
    isVisit_[move] = true;
    return true;
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int totalCost = 0;
    if (size_ != 0)
    {
        for (int i=0; i < size_; i++)
            totalCost += vehicle[i].computeCost(vrp);
    }
    else
    {
        for (int i=0; i < ranSize; i++)
            totalCost += vehicle[i].computeCost(vrp);
    }

    return totalCost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < size_; i++)
    {
        printf("vehicle %2d", i);
        vehicle[i].print();
    }
}
