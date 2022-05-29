#pragma once
#include"sgm_type.h"
#include<omp.h>
#define __builtin_popcountll __popcnt

/**
 * \brief census�任
 * \param source	���룬Ӱ������
 * \param census	�����censusֵ����
 * \param width		���룬Ӱ���
 * \param height	���룬Ӱ���
 */
void census_transform_5x5(const uint8* source, uint32* census, const sint32& width, const sint32& height);

// ���������Բ��ұ����һ����һ��
// ����Census�任��ô�������ֵ���ǳ��ļ򵥣����Ǽ�������censusֵ�ĺ�����hamming�����룬Ҳ��������λ���в�ͬ��λ�ĸ���.
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
