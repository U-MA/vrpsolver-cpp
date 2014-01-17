#ifndef VRPSOLVER_CPP_DEVICEVRP_H
#define VRPSOLVER_CPP_DEVICEVRP_H

vrp_problem *createVrpOnDevice(void);
void deleteVrpOnDevice(vrp_problem *device_vrp);

void copyVrpHostToDevice(vrp_problem *device_vrp, const vrp_problem *host_vrp);


/* usage example
 *
 *
 * int main(int argc, char **argv)
 * {
 *      char file_path = // set file path
 *
 *      vrp_problem *host_vrp = NULL;
 *      vrp_io(host_vrp, file_path);
 *
 *      vrp_problem *device_vrp = createVrpOnDevice();
 *      copyVrpHostToDevice(device_vrp, host_vrp);
 *
 *      // use device_vrp
 *
 *      deleteVrpOnDeice(device_vrp);
 *
 *      return 0;
 *}
 */

#endif /* VRPSOLVER_CPP_DEVICEVRP_H */
