extern "C"
{
#include "vrp_types.h"
}

#include "MonteCarloTreeSearch.h"
#include "Node.h"
#include "VehicleManager.h"

void MCTS::MonteCarloTreeSearch(const vrp_problem *vrp, const VehicleManager& vm)
{
    int visitedSize = 0;
    Node *visited = new Node[200]; /* 訪問したノードを記憶 */

    VehicleManager vm_copy;
    vm_copy.copy(vm); /* 引数vmのコピーを作成 */
    Vehilce v = vm_copy.runVehicle(); /* 現在走行しているVehicleを取得
                                       * (暗黙にシーケンシャルverを想定している */
    /* SELECTION */
    while (!node.isLeaf())
    {
        visited[visitedSize++] = node;
        v.visit(vrp, node.customer());
        node = node.select();
    }

    /* EXPANSION */
    node.expand(vrp->vertnum);
    node = node.select();
    visited[visitedSize++] = node;
    v.visit(vrp, node.customer());

    /* SIMULATION */
    int cost = VrpSimulation::sequentialRandomSimulation(vrp, vm);

    /* BACKPROPAGATION */
    for (int i=0; i < visitedSize; i++)
        visited[i]->update(cost);

    delete[] visited;
}
