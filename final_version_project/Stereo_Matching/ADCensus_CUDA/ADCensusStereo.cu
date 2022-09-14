#include "ADCensusStereo.h"
#include <algorithm>
#include <chrono>
using namespace std::chrono;

ADCensusStereo::ADCensusStereo() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
is_initialized_(false) { }

ADCensusStereo::~ADCensusStereo()
{
	Release();
	is_initialized_ = false;
}

bool ADCensusStereo::Initialize(const sint32& width, const sint32& height, const ADCensusOption& option)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// 算法参数
	option_ = option;

	if (width <= 0 || height <= 0) {
		return false;
	}

	//··· 开辟内存空间
	const sint32 img_size = width_ * height_;
	const sint32 disp_range = option_.max_disparity - option_.min_disparity;
	if (disp_range <= 0) {
		return false;
	}

	// 视差图
	disp_left_ = new float32[img_size];
	disp_right_ = new float32[img_size];

	// 初始化代价计算器
	if (!cost_computer_.Initialize(width_, height_, option_.min_disparity, option_.max_disparity)) {
		is_initialized_ = false;
		return is_initialized_;
	}

	// 初始化代价聚合器
	if (!aggregator_.Initialize(width_, height_, option_.min_disparity, option_.max_disparity)) {
		is_initialized_ = false;
		return is_initialized_;
	}

	// 初始化多步优化器
	if (!refiner_.Initialize(width_, height_)) {
		is_initialized_ = false;
		return is_initialized_;
	}

	is_initialized_ = disp_left_ && disp_right_;

	return is_initialized_;
}

bool ADCensusStereo::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
	if (!is_initialized_) {
		return false;
	}
	if (img_left == nullptr || img_right == nullptr || disp_left == nullptr) {
		return false;
	}

	img_left_ = img_left;
	img_right_ = img_right;

	auto start = steady_clock::now();

	// 代价计算
	ComputeCost();

	auto end = steady_clock::now();
	auto tt = duration_cast<milliseconds>(end - start);
#if LOG
	printf("代价计算耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG

	start = steady_clock::now();

	// 代价聚合
	CostAggregation();

	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);
#if LOG
	printf("代价聚合耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG


	start = steady_clock::now();

	// 扫描线优化
	ScanlineOptimize();

	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);

#if LOG
	printf("扫描线优化耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG
	start = steady_clock::now();

	// 计算左右视图视差
	ComputeDisparity();
	ComputeDisparityRight();

	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);
#if LOG
	printf("视差计算耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG

	start = steady_clock::now();

	// 多步骤视差优化
	MultiStepRefine();

	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);
#if LOG
	printf("视差优化耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG

	start = steady_clock::now();

	// 输出视差图
	memcpy(disp_left, disp_left_, height_ * width_ * sizeof(float32));

	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);

#if LOG
	printf("输出视差耗时 :	%lf s\n", tt.count() / 1000.0);
#endif // LOG
	return true;
}

bool ADCensusStereo::Reset(const uint32& width, const uint32& height, const ADCensusOption& option)
{
	// 释放内存
	Release();

	// 重置初始化标记
	is_initialized_ = false;

	// 初始化
	return Initialize(width, height, option);
}


void ADCensusStereo::ComputeCost()
{
	// 设置代价计算器数据
	cost_computer_.SetData(img_left_, img_right_);
	// 设置代价计算器参数
	cost_computer_.SetParams(option_.lambda_ad, option_.lambda_census);
	// 计算代价
	cost_computer_.Compute();
}

void ADCensusStereo::CostAggregation()
{
	// 设置聚合器数据
	aggregator_.SetData(img_left_, img_right_, cost_computer_.get_cost_ptr());
	// 设置聚合器参数
	aggregator_.SetParams(option_.cross_L1, option_.cross_L2, option_.cross_t1, option_.cross_t2);
	// 代价聚合
	aggregator_.Aggregate(4);
}

void ADCensusStereo::ScanlineOptimize()
{
	// 设置优化器数据
	scan_line_.SetData(img_left_, img_right_, cost_computer_.get_cost_ptr(), aggregator_.get_cost_ptr());
	// 设置优化器参数
	scan_line_.SetParam(width_, height_, option_.min_disparity, option_.max_disparity, option_.so_p1, option_.so_p2, option_.so_tso);
	// 扫描线优化
	scan_line_.Optimize();
}

void ADCensusStereo::MultiStepRefine()
{
	// 设置多步优化器数据
	refiner_.SetData(img_left_, aggregator_.get_cost_ptr(), aggregator_.get_arms_ptr(), disp_left_, disp_right_);
	// 设置多步优化器参数
	refiner_.SetParam(option_.min_disparity, option_.max_disparity, option_.irv_ts, option_.irv_th, option_.lrcheck_thres,
		option_.do_lr_check, option_.do_filling, option_.do_filling, option_.do_discontinuity_adjustment);
	// 多步优化
	refiner_.Refine();
}
__global__ void ComputeDisparityCuda(float32* cost_ptr, float32* disparity,
	const sint32 height, const sint32 width, sint32 max_disparity, sint32 min_disparity)
{

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	auto disp_range = max_disparity - min_disparity;
	if (idx < width && idy < height)
	{
		//float32* cost_local = (float32*)malloc(sizeof(float32) * disp_range);
		float32 min_cost = Large_Float;
		float32 sec_min_cost = Large_Float;
		sint32 best_disparity = 0;

		// ---遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
		for (sint32 d = min_disparity; d < max_disparity; d++) {
			const sint32 d_idx = d - min_disparity;
			//const auto cost = cost_local[d_idx] = cost_ptr[idy * width * disp_range + idx * disp_range + d_idx];
			const auto cost= cost_ptr[idy * width * disp_range + idx * disp_range + d_idx];
			if (min_cost > cost) {
				min_cost = cost;
				best_disparity = d;
			}
		}

#if is_check_unique
		if (is_check_unique) {
			// 再遍历一次，输出次最小代价值
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				if (d == best_disparity) {
					// 跳过最小代价值
					continue;
				}
				const auto& cost = cost_ptr[idy * width * disp_range + idx * disp_range + d];
				sec_min_cost = MIN(sec_min_cost, cost);
			}

			// 判断唯一性约束
			// 若(min-sec)/min < min*(1-uniquness)，则为无效估计
			if (sec_min_cost - min_cost <= static_cast<float32>(min_cost * (1 - uniqueness_ratio))) {
				disparity[idy * width + idx] = Invalid_Float;
				return;
			}
		}
#endif
		// ---子像素拟合
		if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
			disparity[idy * width + idx] = Invalid_Float;
			return;
		}
		// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
		const sint32 idx_1 = best_disparity - 1 - min_disparity;
		const sint32 idx_2 = best_disparity + 1 - min_disparity;
		const float32 cost_1 = cost_ptr[idy * width * disp_range + idx * disp_range + idx_1];
		const float32 cost_2 = cost_ptr[idy * width * disp_range + idx * disp_range + idx_2]; 
		// 解一元二次曲线极值
		const float32 denom = cost_1 + cost_2 - 2 * min_cost;
		if (denom != 0.0f) {
			disparity[idy * width + idx] = static_cast<float32>(best_disparity) + (cost_1 - cost_2) / (denom * 2.0f);
		}
		else {
			disparity[idy * width + idx] = static_cast<float32>(best_disparity);
		}
		//free(cost_local);
	}

}



void ADCensusStereo::ComputeDisparity()
{
	const sint32& min_disparity = option_.min_disparity;
	const sint32& max_disparity = option_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// 左影像视差图
	const auto disparity = disp_left_;
	// 左影像聚合代价数组
	const auto cost_ptr = aggregator_.get_cost_ptr();

	const sint32 width = width_;
	const sint32 height = height_;
	auto img_size = width * height;

	

	// ---逐像素计算最优视差
//#pragma omp parallel for 
//	for (sint32 i = 0; i < height; i++) {
//		// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
//		std::vector<float32> cost_local(disp_range);
//		for (sint32 j = 0; j < width; j++) {
//			float32 min_cost = Large_Float;
//			sint32 best_disparity = 0;
//
//			// ---遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
//			for (sint32 d = min_disparity; d < max_disparity; d++) {
//				const sint32 d_idx = d - min_disparity;
//				const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + j * disp_range + d_idx];
//				if (min_cost > cost) {
//					min_cost = cost;
//					best_disparity = d;
//				}
//			}
//			// ---子像素拟合
//			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
//				disparity[i * width + j] = Invalid_Float;
//				continue;
//			}
//			// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
//			const sint32 idx_1 = best_disparity - 1 - min_disparity;
//			const sint32 idx_2 = best_disparity + 1 - min_disparity;
//			const float32 cost_1 = cost_local[idx_1];
//			const float32 cost_2 = cost_local[idx_2];
//			// 解一元二次曲线极值
//			const float32 denom = cost_1 + cost_2 - 2 * min_cost;
//			if (denom != 0.0f) {
//				disparity[i * width + j] = static_cast<float32>(best_disparity) + (cost_1 - cost_2) / (denom * 2.0f);
//			}
//			else {
//				disparity[i * width + j] = static_cast<float32>(best_disparity);
//			}
//		}
//	}

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

	float32* cost_ptr_cuda = nullptr;
	float32* disparity_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_ptr_cuda, sizeof(float32) * img_size * disp_range));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&disparity_cuda, sizeof(float32) * img_size));


	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_ptr_cuda, cost_ptr, sizeof(float32) * img_size * disp_range, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disparity_cuda, disparity, sizeof(float32) * img_size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());



	// ---逐像素计算最优视差
	ComputeDisparityCuda << <blocksPerGrid, threadsPerBlock >> > (cost_ptr_cuda, disparity_cuda, height, width,
		max_disparity, min_disparity);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disparity, disparity_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaFree(cost_ptr_cuda);
	cudaFree(disparity_cuda);

}

void ADCensusStereo::ComputeDisparityRight()
{
	const sint32& min_disparity = option_.min_disparity;
	const sint32& max_disparity = option_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// 右影像视差图
	const auto disparity = disp_right_;
	// 左影像聚合代价数组
	const auto cost_ptr = aggregator_.get_cost_ptr();

	const sint32 width = width_;
	const sint32 height = height_;

	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
	std::vector<float32> cost_local(disp_range);

	// ---逐像素计算最优视差
	// 通过左影像的代价，获取右影像的代价
	// 右cost(xr,yr,d) = 左cost(xr+d,yl,d)
#pragma omp parallel for 
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			float32 min_cost = Large_Float;
			sint32 best_disparity = 0;

			// ---统计候选视差下的代价值
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				const sint32 d_idx = d - min_disparity;
				const sint32 col_left = j + d;
				if (col_left >= 0 && col_left < width) {
					const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + col_left * disp_range + d_idx];
					if (min_cost > cost) {
						min_cost = cost;
						best_disparity = d;
					}
				}
				else {
					cost_local[d_idx] = Large_Float;
				}
			}

			// ---子像素拟合
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = best_disparity;
				continue;
			}

			// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
			const sint32 idx_1 = best_disparity - 1 - min_disparity;
			const sint32 idx_2 = best_disparity + 1 - min_disparity;
			const float32 cost_1 = cost_local[idx_1];
			const float32 cost_2 = cost_local[idx_2];
			// 解一元二次曲线极值
			const float32 denom = cost_1 + cost_2 - 2 * min_cost;
			if (denom != 0.0f) {
				disparity[i * width + j] = static_cast<float32>(best_disparity) + (cost_1 - cost_2) / (denom * 2.0f);
			}
			else {
				disparity[i * width + j] = static_cast<float32>(best_disparity);
			}
		}
	}
}

void ADCensusStereo::Release()
{
	SAFE_DELETE(disp_left_);
	SAFE_DELETE(disp_right_);
}

