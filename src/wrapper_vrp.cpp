#include <stdlib.h>
#include <string.h>

extern "C"
{
#include "vrp_io.h"
}

#include "wrapper_vrp.h"

/* filename中のkに続く数字文字列を取り出し、整数値に変換 */
static int extractVehicleSizeAndToInt(char *filename)
{
    char *k   = strrchr(filename, 'k');
    char *dot = strrchr(filename, '.');
    int  n    = (dot-k) / sizeof(char);

    char vehicle_size[3];
    strncpy(vehicle_size, k+1, n);
    vehicle_size[n+1] = '\0';
    return atoi(vehicle_size);
}

vrp_problem *createVrpFromFilePath(char *infile)
{
    vrp_problem *vrp = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(vrp, infile);
    vrp->numroutes = extractVehicleSizeAndToInt(infile);
    return vrp;
}

void destroyVrp(vrp_problem *vrp)
{
    if (vrp->demand      != 0) free(vrp->demand);
    if (vrp->posx        != 0) free(vrp->posx);
    if (vrp->posy        != 0) free(vrp->posy);
    if (vrp->dist.cost   != 0) free(vrp->dist.cost);
    if (vrp->dist.coordx != 0) free(vrp->dist.coordx);
    if (vrp->dist.coordy != 0) free(vrp->dist.coordy);
    
    free(vrp); vrp = NULL;
}
