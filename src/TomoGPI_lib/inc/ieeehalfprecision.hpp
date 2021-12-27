/*
 * ieeehalfprecision.h
 *
 *  Created on: 22 sept. 2015
 *      Author: gac
 */

#ifndef IEEEHALFPRECISION_H_
#define IEEEHALFPRECISION_H_



#include <string.h>
#include <stdint.h>

#define INT16_TYPE int16_t
#define UINT16_TYPE uint16_t
#define INT32_TYPE int32_t
#define UINT32_TYPE uint32_t



// Prototypes -----------------------------------------------------------------

int singles2halfp(void *target, void *source, int numel);
int doubles2halfp(void *target, void *source, int numel);
int halfp2singles(void *target, void *source, int numel);
int halfp2doubles(void *target, void *source, int numel);


#endif /* IEEEHALFPRECISION_H_ */
