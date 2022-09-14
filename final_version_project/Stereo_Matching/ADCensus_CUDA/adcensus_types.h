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


/** \brief �������ͱ��� */
typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

typedef uint64			cost_t;

/** \brief float32��Чֵ */
constexpr auto Invalid_Float = std::numeric_limits<float32>::infinity();

constexpr auto Large_Float = 99999.0f;
constexpr auto Small_Float = -99999.0f;

/** \brief Census���ڳߴ����� */
enum CensusSize {
	Census5x5 = 0,
	Census9x7
};

/** \brief ADCensus�����ṹ�� */
struct ADCensusOption {
	sint32  min_disparity;		// ��С�Ӳ�
	sint32	max_disparity;		// ����Ӳ�

	sint32	lambda_ad;			// ����AD����ֵ�Ĳ���
	sint32	lambda_census;		// ����Census����ֵ�Ĳ���
	sint32	cross_L1;			// ʮ�ֽ��洰�ڵĿռ��������L1
	sint32  cross_L2;			// ʮ�ֽ��洰�ڵĿռ��������L2
	sint32	cross_t1;			// ʮ�ֽ��洰�ڵ���ɫ�������t1
	sint32  cross_t2;			// ʮ�ֽ��洰�ڵ���ɫ�������t2
	float32	so_p1;				// ɨ�����Ż�����p1
	float32	so_p2;				// ɨ�����Ż�����p2
	sint32	so_tso;				// ɨ�����Ż�����tso
	sint32	irv_ts;				// Iterative Region Voting������ts
	float32 irv_th;				// Iterative Region Voting������th
	
	float32	lrcheck_thres;		// ����һ����Լ����ֵ

	bool	do_lr_check;					// �Ƿ�������һ����
	bool	do_filling;						// �Ƿ����Ӳ����
	bool	do_discontinuity_adjustment;	// �Ƿ���������������
	
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
* \brief ��ɫ�ṹ��
*/
struct ADColor {
	uint8 r, g, b;
	ADColor() : r(0), g(0), b(0) {}
	ADColor(uint8 _b, uint8 _g, uint8 _r) {
		r = _r; g = _g; b = _b;
	}
};

#endif
