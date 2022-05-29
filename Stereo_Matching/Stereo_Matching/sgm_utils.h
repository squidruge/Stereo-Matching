#pragma once
#include<omp.h>
#define __builtin_popcountll __popcnt

/**
 * \brief census变换
 * \param source	输入，影像数据
 * \param census	输出，census值数组
 * \param width		输入，影像宽
 * \param height	输入，影像高
 */
void census_transform_5x5(const uint8* source, uint32* census, const sint32& width, const sint32& height);

// 可以再试试查找表加速一下这一步
// 基于Census变换怎么计算代价值，非常的简单，就是计算两个census值的汉明（hamming）距离，也就是两个位串中不同的位的个数.
uint8 hamming_distance(const uint32 x, const uint32 y);

void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward);

void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward);

void CostAggregateDagonal_1(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward);

void CostAggregateDagonal_2(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward);
