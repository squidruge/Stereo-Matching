#ifndef ADCENSUS_STEREO_TYPES_H_
#define ADCENSUS_STEREO_TYPES_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <omp.h>
#include "config.h"
#include <cstdint>
#include <limits>
#include <vector>
using std::vector;
using std::pair;

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

#define PI 3.1415926f


/** \brief 基础类型别名 */
typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

typedef uint64			cost_t;

/** \brief float32无效值 */
constexpr auto Invalid_Float = std::numeric_limits<float32>::infinity();

constexpr auto Large_Float = 99999.0f;
constexpr auto Small_Float = -99999.0f;

/** \brief Census窗口尺寸类型 */
enum CensusSize {
	Census5x5 = 0,
	Census9x7
};

/** \brief ADCensus参数结构体 */
struct ADCensusOption {
	sint32  min_disparity;		// 最小视差
	sint32	max_disparity;		// 最大视差

	sint32	lambda_ad;			// 控制AD代价值的参数
	sint32	lambda_census;		// 控制Census代价值的参数
	sint32	cross_L1;			// 十字交叉窗口的空间域参数：L1
	sint32  cross_L2;			// 十字交叉窗口的空间域参数：L2
	sint32	cross_t1;			// 十字交叉窗口的颜色域参数：t1
	sint32  cross_t2;			// 十字交叉窗口的颜色域参数：t2
	float32	so_p1;				// 扫描线优化参数p1
	float32	so_p2;				// 扫描线优化参数p2
	sint32	so_tso;				// 扫描线优化参数tso
	sint32	irv_ts;				// Iterative Region Voting法参数ts
	float32 irv_th;				// Iterative Region Voting法参数th
	
	float32	lrcheck_thres;		// 左右一致性约束阈值

	bool	do_lr_check;					// 是否检查左右一致性
	bool	do_filling;						// 是否做视差填充
	bool	do_discontinuity_adjustment;	// 是否做非连续区调整
	
	ADCensusOption(): min_disparity(0), max_disparity(64),
	                  lambda_ad(LAMBDA_AD), lambda_census(LAMBDA_CENSUS),
	                  cross_L1(CrossL1), cross_L2(CrossL2),
	                  cross_t1(CrossTau1), cross_t2(CrossTau2),
	                  so_p1(1.0f), so_p2(3.0f),
	                  so_tso(15), irv_ts(20), irv_th(0.4f),
	                  lrcheck_thres(1.0f),
					  do_lr_check(true), do_filling(true), do_discontinuity_adjustment(true) {} ;


};

/**
* \brief 颜色结构体
*/
struct ADColor {
	uint8 r, g, b;
	ADColor() : r(0), g(0), b(0) {}
	ADColor(uint8 _b, uint8 _g, uint8 _r) {
		r = _r; g = _g; b = _b;
	}
};

#endif
