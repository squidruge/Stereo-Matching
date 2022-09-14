#ifndef COSTS_H_
#define COSTS_H_

#include <stdint.h>
#include "adcensus_types.h"
#include "config.h"
//#include "configuration.h"

__global__ void CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, cost_t *transform, cost_t *transform2, const uint32_t rows, const uint32_t cols);

#endif /* COSTS_H_ */
