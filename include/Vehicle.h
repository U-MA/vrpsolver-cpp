#ifndef VRPSOLVER_CPP_VEHICLE_H
#define VRPSOLVER_CPP_VEHICLE_H

extern "C"
{
#include "vrp_types.h"
}

#define OUT_OF_BOUND -1

class Vehicle
{
public:
    Vehicle(void);
    Vehicle(const Vehicle& v);
    ~Vehicle(void);

    /* メンバ変数を初期化する */
    void init(void);

    /* routeの中身が空かどうか */
    bool empty(void) const;

    /* accessor */
    bool isVisit(int customer) const;
    int  quantity(void) const;

    bool visit(const vrp_problem *vrp, int customer);
    int  computeCost(const vrp_problem *vrp) const;

    void print(void) const;
    Vehicle copy() const;

private:
    static const int MAXSIZE = 130;
    bool isVisit_[MAXSIZE];
    int  route[MAXSIZE];
    int  routeSize;
    int  quantity_;
};

#endif /* VRPSOLVER_CPP_VEHICLE_H */
