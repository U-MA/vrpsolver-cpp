#include <cstring>
#include <cstdlib>

extern "C"
{
#include "vrp_io.h"
}

#include "host_vrp.h"

HostVrp::HostVrp(void)
{
    vrp_ = (vrp_problem *)calloc(1, sizeof(vrp_problem));
}

/* filename中のkに続く数字文字列を取り出し、整数値に変換 */
static int extractVehicleSizeAndToInt(const char *filename)
{
    char *k   = strrchr(filename, 'k');
    char *dot = strrchr(filename, '.');
    long n    = (dot-k) / sizeof(char);

    char vehicle_size[3];
    strncpy(vehicle_size, k+1, n);
    vehicle_size[n+1] = '\0';
    return atoi(vehicle_size);
}

HostVrp::HostVrp(const char *file_path)
{
    vrp_ = (vrp_problem *)calloc(1, sizeof(vrp_problem));
    vrp_io(vrp_, file_path);
    vrp_->numroutes = extractVehicleSizeAndToInt(file_path);
}

HostVrp::~HostVrp(void)
{
    if (vrp_->demand      != 0) free(vrp_->demand);
    if (vrp_->posx        != 0) free(vrp_->posx);
    if (vrp_->posy        != 0) free(vrp_->posy);
    if (vrp_->dist.cost   != 0) free(vrp_->dist.cost);
    if (vrp_->dist.coordx != 0) free(vrp_->dist.coordx);
    if (vrp_->dist.coordy != 0) free(vrp_->dist.coordy);
    
    free(vrp_); vrp_ = NULL;
}


void HostVrp::Create(const char *file_path)
{
    vrp_io(vrp_, file_path);
    vrp_->numroutes = extractVehicleSizeAndToInt(file_path);
}

const char *HostVrp::name(void) const
{
    const char *aName = vrp_->name;
    return aName;
}
