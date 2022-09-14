#ifndef AD_CENSUS_SCANLNE_OPTIMIZER_H_
#define AD_CENSUS_SCANLNE_OPTIMIZER_H_

#include <algorithm>

#include "adcensus_types.h"
#include "config.h"
#include "MemoryFunction.h"
#include "adcensus_types.h"


/* 扫描线优化核函数 */
__global__ void ScanlineOptimizeLeftRight(const uint8* img_left, const uint8* img_right,
	const float32* cost_so_src, float32* cost_so_dst, bool is_forward,
	const sint32 width, const sint32 height, const sint32  min_disparity,
	const sint32 max_disparity, const float32  p1, const float32  p2, const sint32  tso);

__global__ void ScanlineOptimizeUpDown(const uint8* img_left, const uint8* img_right,
	const float32* cost_so_src, float32* cost_so_dst, bool is_forward,
	const sint32 width, const sint32 height, const sint32  min_disparity,
	const sint32 max_disparity, const float32  p1, const float32  p2, const sint32  tso, const uint32 num);

class ScanlineOptimizer {
public:
	ScanlineOptimizer();
	~ScanlineOptimizer();


	
	void SetData(const uint8* img_left, const uint8* img_right, float32* cost_init, float32* cost_aggr);


	void SetParam(const sint32& width,const sint32& height, const sint32& min_disparity, const sint32& max_disparity, const float32& p1, const float32& p2, const sint32& tso);


	void Optimize();


	
private:

	sint32	width_;
	sint32	height_;


	const uint8* img_left_;
	const uint8* img_right_;
	

	float32* cost_init_;
	float32* cost_aggr_;


	sint32 min_disparity_;

	sint32 max_disparity_;

	float32 so_p1_;

	float32 so_p2_;

	sint32 so_tso_;
};
#endif
