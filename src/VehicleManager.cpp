#include "VehicleManager.h"


VehicleManager::VehicleManager(void)
{
    runVehicle = 0;
}

VehicleManager::VehicleManager(int vehicleSize)
{
    this->vehicleSize = vehicleSize;
    runVehicle = 0;
    for (int i=0; i < CUSTOMER_MAX; i++)
        isVisit[i] = false;
}

VehicleManager::~VehicleManager(void)
{
}

int VehicleManager::getRunningVehicleNumber(void)
{
    return runVehicle;
}

bool VehicleManager::isVisitAll(vrp_problem *vrp)
{
    int customerSize = vrp->vertnum-1;
    for (int i=0; i < customerSize; i++)
        if (!isVisit[i]) return false;

    return true;
}

bool VehicleManager::changeVehicle(void)
{
    if (vehicleSize <= runVehicle+1) return false;

    runVehicle++;
    return true;
}

bool VehicleManager::update(vrp_problem *vrp, int customer)
{
    if (vehicle[runVehicle].visit(vrp, customer))
        return (isVisit[customer-1] = true);

    return false;
}

int VehicleManager::computeTotalCost(vrp_problem *vrp)
{
    int totalCost = 0;
    for (int i=0; i < vehicleSize; i++)
        totalCost += vehicle[i].computeCost(vrp);

    return totalCost;
}

void VehicleManager::print(void)
{
    for (int i=0; i < vehicleSize; i++)
    {
        printf("vehicle %2d", i);
        vehicle[i].print();
    }
}

int VehicleManager::randomSimulation(vrp_problem *vrp)
{
    int candidates[CUSTOMER_MAX], candidatesSize;

    /* 全ての顧客を訪問するか、全ての車体を使いきるまで繰り返す */
    while (!isVisitAll(vrp) && vehicleSize > runVehicle)
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
