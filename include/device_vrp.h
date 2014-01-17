#ifndef VRPSOLVER_CPP_DEVICEVRP_H
#define VRPSOLVER_CPP_DEVICEVRP_H

vrp_problem *createVrpOnDevice(void);
void transferVrpHostToDevice(vrp_problem *device_vrp, const vrp_problem *host_vrp);

#endif /* VRPSOLVER_CPP_DEVICEVRP_H */
