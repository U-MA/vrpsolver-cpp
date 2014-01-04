#include "CppUTest/TestHarness.h"

extern "C"
{
#include "vrp_types.h"
#include "vrp_macros.h"
}

#include "Node.h"
#include "VehicleManager.h"
#include "Vehicle.h"

TEST_GROUP(acceptTest)
{
    vrp_problem *vrp;

    void setup()
    {
        vrp = (vrp_problem *)malloc(sizeof(vrp_problem));
        vrp->dist.cost = (int *)calloc(100, sizeof(int));
        vrp->demand    = (int *)calloc(100, sizeof(int));
    }

    void teardown()
    {
        free(vrp->demand);
        free(vrp->dist.cost);
        free(vrp);
    }

    void Vrp_SetCost(int first, int second, int value)
    {
        vrp->dist.cost[INDEX(first, second)] = value;
    }

    /* Applying to Monte Carlo Techniques to the Capacitated Vehicle
     * Routing Problem Table 2.1, 2.2より */
    void Vrp_SetProblem(void)
    {
        vrp->vertnum = 6;
        vrp->edgenum = vrp->vertnum * (vrp->vertnum-1) / 2;

        Vrp_SetCost(0, 1, 28);
        Vrp_SetCost(0, 2, 31);
        Vrp_SetCost(0, 3, 20);
        Vrp_SetCost(0, 4, 25);
        Vrp_SetCost(0, 5, 34);
        Vrp_SetCost(1, 2, 21);
        Vrp_SetCost(1, 3, 29);
        Vrp_SetCost(1, 4, 26);
        Vrp_SetCost(1, 5, 20);
        Vrp_SetCost(2, 3, 38);
        Vrp_SetCost(2, 4, 20);
        Vrp_SetCost(2, 5, 32);
        Vrp_SetCost(3, 4, 30);
        Vrp_SetCost(3, 5, 27);
        Vrp_SetCost(4, 5, 25);

        vrp->capacity  = 100;
        vrp->demand[1] = 37;
        vrp->demand[2] = 35;
        vrp->demand[3] = 30;
        vrp->demand[4] = 25;
        vrp->demand[5] = 32;
    }
};

IGNORE_TEST(acceptTest, MonteCarloTreeSearch)
{
    VehicleManager vm;
    Vehicle        v;

    v.init();

    while (!vm.isVisitAll(vrp))
    {
        Node mct;

        /* 好きなだけイテレーションさせる
         * 回数や時間を用いて上限を決める */
        for(int i=0; i < 1000; i++)
            mct.search(vrp, vm, v);

        /* イテレーションした結果, 一番有望な手を選択 */
        int move = mct.next();

        if (move == 0)
        {
            /* 次の車体に変更 */
            vm.add(v);
            v.init();

            /* 用意されている車体を使いきったか確認 */
            if (vm.size()+1 >= vrp->numroutes)
                break;
        }
        else
        {
            /* 顧客を訪問する */
            if (!v.visit(vrp, move))
            {
                printf("visit error\n");
                exit(1);
            }
        }
    }

    int cost = 1e6;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    CHECK(cost > 0);
}
