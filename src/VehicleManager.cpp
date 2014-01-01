#include "VehicleManager.h"


VehicleManager::VehicleManager(void)
{
    runVehicle = 0;
}

VehicleManager::VehicleManager(int size)
{
    this->size = size;
    runVehicle = 0;
    for (int i=0; i < CUSTOMER_MAX; i++)
        isVisit[i] = false;
}

VehicleManager::~VehicleManager(void)
{
}

int VehicleManager::getRunningVehicleNumber(void) const
{
    return runVehicle;
}

int VehicleManager::getEmptyVehicle(void) const
{
    for (int i=0; i < size; i++)
    {
        if (vehicle[i].empty())
            return i;
    }
    return -1;
}

Vehicle VehicleManager::getVehicle(int id)
{
    return vehicle[id];
}

bool VehicleManager::empty(void)
{
    return size == 0;
}

bool VehicleManager::isVisitAll(const vrp_problem *vrp) const
{
    int customerSize = vrp->vertnum-1;
    for (int i=0; i < customerSize; i++)
        if (!isVisit[i]) return false;

    return true;
}

bool VehicleManager::isVisitOne(int customer) const
{
    return isVisit[customer-1];
}

bool VehicleManager::changeVehicle(void)
{
    if (size <= runVehicle+1) return false;

    runVehicle++;
    return true;
}

bool VehicleManager::update(const vrp_problem *vrp, int customer)
{
    if (vehicle[runVehicle].visit(vrp, customer))
        return (isVisit[customer-1] = true);

    return false;
}

int VehicleManager::computeTotalCost(const vrp_problem *vrp) const
{
    int totalCost = 0;
    for (int i=0; i < size; i++)
        totalCost += vehicle[i].computeCost(vrp);

    return totalCost;
}

void VehicleManager::print(void) const
{
    for (int i=0; i < size; i++)
    {
        printf("vehicle %2d", i);
        vehicle[i].print();
    }
}

int VehicleManager::randomSimulation(const vrp_problem *vrp)
{
    int candidates[CUSTOMER_MAX], candidatesSize;

    /* 全ての顧客を訪問するか、全ての車体を使いきるまで繰り返す */
    while (!isVisitAll(vrp) && size > runVehicle)
    {
        candidatesSize = 0;
        /* 次に訪れる顧客の候補を調べる */
        for (int i=1; i < vrp->vertnum; i++)
        {
            if (!isVisit[i-1] && (vehicle[runVehicle].getQuantity() + vrp->demand[i] <= vrp->capacity))
            {
                candidates[candidatesSize] = i;
                candidatesSize++;
            }
        }

        if (candidatesSize == 0)
        {
            /* 候補がいなければ次の車体へ変更する */
            changeVehicle();
        }
        else
        {
            /* 候補の中から無作為に一つ選ぶ */
            int nextCustomer = candidates[rand() % candidatesSize];
            if (!update(vrp, nextCustomer))
            {
                printf("vehicle %d can not visit customer %d\n", runVehicle, nextCustomer);
                exit(1);
            }
        }
    }

    /* 全ての顧客を訪問していればコストを計算 */
    int cost = 1000000;
    if (isVisitAll(vrp))
        cost = computeTotalCost(vrp);

    return cost;
}
