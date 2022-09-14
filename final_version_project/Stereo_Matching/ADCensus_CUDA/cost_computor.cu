#include "cost_computor.h"
#include "adcensus_types.h"
#include "census_cost.h"
#include <iostream>

CostComputor::CostComputor() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
lambda_ad_(0), lambda_census_(0), min_disparity_(0), max_disparity_(0),
gray_left_(nullptr), gray_right_(nullptr),
census_left_(nullptr), census_right_(nullptr),
cost_init_(nullptr),
is_initialized_(false) { }

CostComputor::~CostComputor()
{

}

bool CostComputor::Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;

	// cudaStream_t stream1, stream2, stream3;
	// Create streams

	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));

	const sint32 img_size = width_ * height_;
	const sint32 disp_range = max_disparity_ - min_disparity_;
	if (img_size <= 0 || disp_range <= 0) {
		is_initialized_ = false;
		return false;
	}



	// 灰度数据（左右影像）
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&gray_left_, sizeof(uint8_t) * img_size));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&gray_right_, sizeof(uint8_t) * img_size));


	// census数据（左右影像）

	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&census_left_, sizeof(uint64) * img_size));
	CUDA_CHECK_RETURN(cudaMallocManaged((void**)&census_right_, sizeof(uint64) * img_size));



	// 初始代价数据
	cost_init_ = (float32*)malloc(sizeof(float32) * img_size * disp_range);


	is_initialized_ = gray_left_ && gray_right_ && census_left_ && census_right_ && cost_init_;
	return is_initialized_;
}

void CostComputor::SetData(const uint8* img_left, const uint8* img_right)
{

	img_left_ = img_left;
	img_right_ = img_right;
}

void CostComputor::SetParams(const sint32& lambda_ad, const sint32& lambda_census)
{
	lambda_ad_ = lambda_ad;
	lambda_census_ = lambda_census;
}

__global__ void ComputeGrayCuda(const uint8* img,  uint8* img_gray,
	const sint32 height, const sint32 width)
{

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < width && idy < height)
	{
		const auto b = img[idy * width * 3 + 3 * idx];
		const auto g = img[idy * width * 3 + 3 * idx + 1];
		const auto r = img[idy * width * 3 + 3 * idx + 2];
		img_gray[idy * width + idx] = uint8(r * 0.299 + g * 0.587 + b * 0.114);
	}

}

void CostComputor::ComputeGray()
{
	 //彩色转灰度
	const sint32 img_size = width_ * height_;

	uint8_t* img_left_cuda;
	uint8_t* img_right_cuda;
	
	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_left_cuda, sizeof(uint8_t) * img_size * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_right_cuda, sizeof(uint8_t) * img_size * 3));

	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_left_cuda, img_left_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_right_cuda, img_right_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));//stream2?


	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

	ComputeGrayCuda << <blocksPerGrid, threadsPerBlock, 0, stream1 >> >
		(img_left_cuda, gray_left_, height_, width_);
	ComputeGrayCuda << <blocksPerGrid, threadsPerBlock, 0, stream2 >> >
		(img_right_cuda, gray_right_, height_, width_);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	//释放
	cudaFree(img_left_cuda);
	cudaFree(img_right_cuda);
}

void CostComputor::CensusTransform()
{
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (width_ + block_size.x - 1) / block_size.x;
	grid_size.y = (height_ + block_size.y - 1) / block_size.y;


	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	CenterSymmetricCensusKernelSM2 << <grid_size, block_size, 0, stream1 >> > (gray_left_, gray_right_, census_left_, census_right_, height_, width_);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//cudaEventRecord(stop, 0);
	//float elapsed_time_ms;
	//cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//std::cout << "CensusTransform:" << elapsed_time_ms << "ms" << std::endl;

}

__global__ void ComputeCostCuda(const uint8* img_left, const uint8* img_right, const cost_t* census_left, const cost_t* census_right,
	float32* cost_init, const int height, const int width, sint32 lambda_ad, sint32 lambda_census)
{
	const sint32 disp_range = MAX_DISPARITY - MIN_DISPARITY;
	const sint32 min_disparity = MIN_DISPARITY;
	const sint32 max_disparity = MAX_DISPARITY;

	// 预设参数
	//const auto lambda_ad = lambda_ad;
	//const auto lambda_census = lambda_census;

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	// 计算代价
	if (idy < height && idx < width) {



		const auto bl = img_left[idy * width * 3 + 3 * idx];
		const auto gl = img_left[idy * width * 3 + 3 * idx + 1];
		const auto rl = img_left[idy * width * 3 + 3 * idx + 2];
		const auto& census_val_l = census_left[idy * width + idx];
		// 逐视差计算代价值
		for (sint32 d = min_disparity; d < max_disparity; d++) {
			auto& cost = cost_init[idy * width * disp_range + idx * disp_range + (d - min_disparity)];
			const sint32 xr = idx - d;
			if (xr < 0 || xr >= width) {
				cost = 1.0f;
				continue;
			}

			// ad代价
			const auto br = img_right[idy * width * 3 + 3 * xr];
			const auto gr = img_right[idy * width * 3 + 3 * xr + 1];
			const auto rr = img_right[idy * width * 3 + 3 * xr + 2];
			const float32 cost_ad = (abs(bl - br) + abs(gl - gr) + abs(rl - rr)) / 3.0f;

			// census代价
			const auto& census_val_r = census_right[idy * width + xr];

			uint64 dist = 0, val = census_val_l ^ census_val_r;

			// Count the number of set bits
			while (val) {
				++dist;
				val &= val - 1;
			}


			const float32 cost_census = static_cast<float32>(dist);

			// ad-census代价
			cost = 1 - exp(-cost_ad / lambda_ad) + 1 - exp(-cost_census / lambda_census);
		}

	}
}

void CostComputor::ComputeCost()
{
	const sint32 img_size = width_ * height_;
	auto disp_range = max_disparity_ - min_disparity_;

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	uint8_t* img_left_cuda;
	uint8_t* img_right_cuda;


	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_left_cuda, sizeof(uint8_t) * img_size * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_right_cuda, sizeof(uint8_t) * img_size * 3));

	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_left_cuda, img_left_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_right_cuda, img_right_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));//stream2?

	//设置 block_size
	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	//设置 grid_size
	dim3 grid_size;
	grid_size.x = (width_ + block_size.x - 1) / block_size.x;
	grid_size.y = (height_ + block_size.y - 1) / block_size.y;
	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);


	
	float32* cost_init_cuda=nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_init_cuda, sizeof(float32) * img_size * disp_range));

	CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
	ComputeCostCuda << <grid_size, block_size, 0, stream1 >> > (img_left_cuda,
		img_right_cuda, census_left_, census_right_, cost_init_cuda, height_, width_, lambda_ad_, lambda_census_);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_init_ , cost_init_cuda, sizeof(float32) * img_size * disp_range, cudaMemcpyDeviceToHost, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
																																			 // CUDA FREE
	cudaFree(img_left_cuda);
	cudaFree(img_right_cuda);
	cudaFree(census_left_);
	cudaFree(census_right_);
	cudaFree(cost_init_cuda);

	//cudaEventRecord(stop, 0);
	//
	//float elapsed_time_ms;
	//cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//std::cout << "ComputeCost:" << elapsed_time_ms << "ms" << std::endl;

}

void CostComputor::Compute()
{
	if (!is_initialized_) {
		return;
	}

	// 计算灰度图
	ComputeGray();

	// census变换
	CensusTransform();

	// 代价计算
	ComputeCost();
}

float32* CostComputor::get_cost_ptr()
{
	if (cost_init_ != nullptr) {
		return &cost_init_[0];
	}
	else {
		return nullptr;
	}
}
