extern "C"
{
#include "vrp_types.h"
}

#include "MonteCarloTreeSearch.h"
#include "Node.h"
#include "VehicleManager.h"
#include "VrpSimulation.h"

MonteCarloTree::MonteCarloTree(void)
{
    size_ = 0;
    node  = NULL;
}

MonteCarloTree::~MonteCarloTree(void)
{
}

void MonteCarloTree::init(void)
{
    size_ = 0;
}

static void vehicleUpdate(const vrp_problem *vrp,
                          VehicleManager *vm,
                          Vehicle *v, int move)
{
    if (move == 0)
    {
        vm->add(*v);
        v->init();
    }
    else
        v->visit(vrp, move);
}

void MonteCarloTree::search(const vrp_problem *vrp,
                            const VehicleManager& vm,
                            const Vehicle& v)
{
    printf("in search\n");
    /* 引数として渡されるvm, vは変更しない
     * そのため変更させるための変数を作成 */
    VehicleManager vm_copy = vm.copy();
    Vehicle        v_copy  = v.copy();

    Node *visited[300];
    int  visitedSize = 0;

    Node *root = &node[0];

    /* SELECTION */
    while (!root->isLeaf())
    {
        visited[visitedSize++] = root;
        vehicleUpdate(vrp, &vm_copy, &v_copy, root->customer());
        root = root->select();
    }

    /* EXPANSION */
    /* 顧客の数+1の子を作成. +1は車体の変更 */
    root->expand(vrp->vertnum);
    Node *newNode = root->select();
    visited[visitedSize++] = newNode;
    vehicleUpdate(vrp, &vm_copy, &v_copy, newNode->customer());

    /* SIMULATION */
    int cost = VrpSimulation::sequentialRandomSimulation(vrp, vm_copy, v_copy);

    /* BACKPROPAGATION */
    for (int i=0; i < visitedSize; i++)
        visited[i]->update(cost);
}


int MonteCarloTree::next(void)
{
    return 0;
}
