#include "CppUTest/TestHarness.h"


extern "C"
{
#include "vrp_macros.h"
#include "vrp_types.h"
}

#include "MonteCarloTreeSearch.h"
#include "VehicleManager.h"

TEST_GROUP(MonteCarloTreeSearch)
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

/* モンテカルロ木探索の受け入れテスト的側面のあるテスト
 * モンテカルロ木探索を１回行い、その中で最も有望だと思われる顧客を
 * 返すことを望む */
IGNORE_TEST(MonteCarloTreeSearch, customerTest)
{
}

/* MonteCarloTreeSearchをこう使いたいなぁという簡単な記述
 * 故にテストではないのでこれは残しておくのであれば
 * 常にIGNOREにする */
IGNORE_TEST(MonteCarloTreeSearch, sketch)
{
    /* 問題の設定 */
    Vrp_SetProblem();

    Node mct;
    VehicleManager vm;
    Vehicle v;

    /* 車体の初期化 */
    v.init();

    /* 全ての顧客を訪問するまで続ける */
    while (!vm.isVisitAll(vrp))
    {
        /* モンテカルロ木の初期化 */
        mct.init();

        /* モンテカルロ木を成長させる
         * 好きなだけイテレーションさせる
         * それは回数かもしれないし、時間かもしれない
         * 今は回数にしている */
        for (int i=0; i < 1000; i++)
            mct.search(vrp, vm, v);

        /* モンテカルロ木を成長させた結果
         * 一番有望な手を返す */
        int move = mct.next();

        if (move == 0)
        {
            /* 次の車体に移る */
            vm.add(v);
            v.init();

            /* 用意されてる車体の数を使い切った */
            if (vm.size() == vrp->numroutes)
                break;
        }
        else
            v.visit(vrp, move);
    }

    int cost = 100000;
    if (vm.isVisitAll(vrp))
        cost = vm.computeTotalCost(vrp);

    CHECK(cost > 0);
}
