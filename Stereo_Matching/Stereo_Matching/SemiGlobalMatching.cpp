#include "SemiGlobalMatching.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <chrono>

SemiGlobalMatching::SemiGlobalMatching() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
census_left_(nullptr), census_right_(nullptr),
cost_init_(nullptr), cost_aggr_(nullptr),
cost_aggr_1_(nullptr), cost_aggr_2_(nullptr),
cost_aggr_3_(nullptr), cost_aggr_4_(nullptr),
cost_aggr_5_(nullptr), cost_aggr_6_(nullptr),
cost_aggr_7_(nullptr), cost_aggr_8_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
is_initialized_(false)
{
}


SemiGlobalMatching::~SemiGlobalMatching()
{
	Release();
	is_initialized_ = false;
}

bool SemiGlobalMatching::Initialize(const sint32& width, const sint32& height, const SGMOption& option)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// SGM参数
	option_ = option;

	if (width == 0 || height == 0) {
		return false;
	}

	//··· 开辟内存空间

	// census值（左右影像）
	const sint32 img_size = width * height;
	if (option.census_size == Census5x5) {
		census_left_ = new uint32[img_size]();
		census_right_ = new uint32[img_size]();
	}
	else {
		census_left_ = new uint64[img_size]();
		census_right_ = new uint64[img_size]();
	}

	// 视差范围
	const sint32 disp_range = option.max_disparity - option.min_disparity;
	if (disp_range <= 0) {
		return false;
	}

	// 匹配代价（初始/聚合）
	const sint32 size = width * height * disp_range;
	cost_init_ = new uint8[size]();
	cost_aggr_ = new uint16[size]();
	cost_aggr_1_ = new uint8[size]();
	cost_aggr_2_ = new uint8[size]();
	cost_aggr_3_ = new uint8[size]();
	cost_aggr_4_ = new uint8[size]();
	cost_aggr_5_ = new uint8[size]();
	cost_aggr_6_ = new uint8[size]();
	cost_aggr_7_ = new uint8[size]();
	cost_aggr_8_ = new uint8[size]();

	// 视差图
	disp_left_ = new float32[img_size]();
	disp_right_ = new float32[img_size]();

	is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

	return is_initialized_;
}


void SemiGlobalMatching::Release()
{
	// 释放内存
	SAFE_DELETE(census_left_);
	SAFE_DELETE(census_right_);
	SAFE_DELETE(cost_init_);
	SAFE_DELETE(cost_aggr_);
	SAFE_DELETE(cost_aggr_1_);
	SAFE_DELETE(cost_aggr_2_);
	SAFE_DELETE(cost_aggr_3_);
	SAFE_DELETE(cost_aggr_4_);
	SAFE_DELETE(cost_aggr_5_);
	SAFE_DELETE(cost_aggr_6_);
	SAFE_DELETE(cost_aggr_7_);
	SAFE_DELETE(cost_aggr_8_);
	SAFE_DELETE(disp_left_);
	SAFE_DELETE(disp_right_);
}