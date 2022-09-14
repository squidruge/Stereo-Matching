#include "multistep_refiner.h"
#include "config.h"


//
//__global__ void OutlierDetectionCuda(float32* disp_left, float32* disp_right, const uint32_t width, const uint32_t height, const float32 threshold,
//	bool* is_occlusions, bool* is_mismatches)
//{
//
//	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
//
//
//	// LRCheck 左右一致性
//	if (idy < height && idx < width) {
//
//		is_occlusions[idy * width + idx] = false;// 遮挡区像素
//		is_mismatches[idy * width + idx] = false;//误匹配区像素
//
//		// 左影像视差值
//		auto& disp = disp_left[idy * width + idx];
//
//
//		if (disp == Invalid_Float) {
//
//			is_mismatches[idy * width + idx] = true;
//			return;
//		}
//
//		//当前估计的右图同名点所在列
//		const auto right_pix = lround(idx - disp);
//		if (right_pix >= 0 && right_pix < width) {
//			//右图同名点对应视差
//			const auto& disp_r = disp_right[idy * width + right_pix];
//			// 同名点视差值的差值是否在预设参数内
//			if (abs(disp - disp_r) > threshold) {
//
//				const sint32 col_rl = lround(right_pix + disp_r);
//				if (col_rl > 0 && col_rl < width) {
//					const auto& disp_l = disp_left[idy * width + col_rl];
//					if (disp_l > disp) {
//
//						is_occlusions[idy * width + idx] = true;
//					}
//					else {
//						is_mismatches[idy * width + idx] = true;
//					}
//				}
//				else {
//
//					is_mismatches[idy * width + idx] = true;
//				}
//
//				// 误匹配与遮挡的视差均置为无效状态
//				disp = Invalid_Float;
//			}
//		}
//		else {
//			// 找不到右图中的同名点
//			disp = Invalid_Float;
//
//			is_mismatches[idy * width + idx] = true;
//
//		}
//
//	}
//}

//最小阈值版
__global__ void OutlierDetectionCuda(float32* disp_left, float32* disp_right, const uint32_t width, const uint32_t height, const float32 threshold,
	bool* is_occlusions, bool* is_mismatches)
{

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;


	// ---左右一致性检查
	if (idy < height && idx < width) {

		is_occlusions[idy * width + idx] = false;// 遮挡区像素
		is_mismatches[idy * width + idx] = false;//误匹配区像素

		// 左影像视差值
		auto& disp = disp_left[idy * width + idx];


		if (disp < MinDispMismatch) {

#if MismatchesRefine
			//printf("disp too small\n");
			disp = Invalid_Float;
			is_mismatches[idy * width + idx] = true;

#endif // MismatchesRefine
			return;
		}

		if (disp == Invalid_Float ) {

#if MismatchesRefine
			is_mismatches[idy * width + idx] = true;

#endif // MismatchesRefine
			return;
		}

		// 根据视差值找到右影像上对应的同名像素
		const auto col_right = lround(idx - disp);
		if (col_right >= 0 && col_right < width) {
			// 右影像上同名像素的视差值
			const auto& disp_r = disp_right[idy * width + col_right];
			// 判断两个视差值是否一致（差值在阈值内）
			if (abs(disp - disp_r) > threshold) {
				// 区分遮挡区和误匹配区
				// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl

				const sint32 col_rl = lround(col_right + disp_r);
				if (col_rl > 0 && col_rl < width) {
					const auto& disp_l = disp_left[idy * width + col_rl];
					if (disp_l > disp) {
#if OcclusionsRefine
						is_occlusions[idy * width + idx] = true;
#endif
					}
					else {
#if MismatchesRefine
						is_mismatches[idy * width + idx] = true;
#endif // MismatchesRefine

						
					}
				}
				else {

#if MismatchesRefine
					is_mismatches[idy * width + idx] = true;
#endif // MismatchesRefine
				}

				// 让视差值无效
				disp = Invalid_Float;
			}
		}
		else {
			// 通过视差值在右影像上找不到同名像素（超出影像范围）
			disp = Invalid_Float;

#if MismatchesRefine
			is_mismatches[idy * width + idx] = true;

#endif // MismatchesRefine
		}

	}
}




__global__ void EdgeDetectCuda(uint8* edge_mask, const float32* disp_ptr, const sint32 width, const sint32 height, const float32 threshold)
{

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int idy = blockIdx.y * blockDim.y + threadIdx.y;
	// sobel算子
	if ((idy >= 1 && idy < height - 1) && (idx >= 1 && idx < width - 1)) {

		const auto grad_x = (-disp_ptr[(idy - 1) * width + idx - 1] + disp_ptr[(idy - 1) * width + idx + 1]) +
			(-2 * disp_ptr[idy * width + idx - 1] + 2 * disp_ptr[idy * width + idx + 1]) +
			(-disp_ptr[(idy + 1) * width + idx - 1] + disp_ptr[(idy + 1) * width + idx + 1]);
		const auto grad_y = (-disp_ptr[(idy - 1) * width + idx - 1] - 2 * disp_ptr[(idy - 1) * width + idx] - disp_ptr[(idy - 1) * width + idx + 1]) +
			(disp_ptr[(idy + 1) * width + idx - 1] + 2 * disp_ptr[(idy + 1) * width + idx] + disp_ptr[(idy + 1) * width + idx + 1]);
		const auto grad = abs(grad_x) + abs(grad_y);
		if (grad > threshold) {
			edge_mask[idy * width + idx] = 1;
		}

	}
}


//中值滤波
__global__ void MedianFilter3x3(const float32* d_input, float32* d_out, const uint32_t rows, const uint32_t cols);

template<int n, typename T>
__inline__ __device__ void MedianFilter(const T* d_input, T* d_out, const uint32_t rows, const uint32_t cols) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t row = idx / cols;
	const uint32_t col = idx % cols;
	T window[n * n];
	int half = n / 2;

	if (row >= half && col >= half && row < rows - half && col < cols - half) {
		//return;
		for (uint32_t i = 0; i < n; i++) {
			for (uint32_t j = 0; j < n; j++) {
				window[i * n + j] = d_input[(row - half + i) * cols + col - half + j];
			}
		}

		for (uint32_t i = 0; i < (n * n / 2) + 1; i++) {
			uint32_t min_idx = i;
			for (uint32_t j = i + 1; j < n * n; j++) {
				if (window[j] < window[min_idx]) {
					min_idx = j;
				}
			}
			const T tmp = window[i];
			window[i] = window[min_idx];
			window[min_idx] = tmp;
		}
		d_out[idx] = window[n * n / 2];
	}
	else if (row < rows && col < cols) {
		//return;
		d_out[idx] = d_input[idx];

	}
}

__global__ void MedianFilter3x3(const float32* d_input, float32* d_out, const uint32_t rows, const uint32_t cols) {
	MedianFilter<3>(d_input, d_out, rows, cols);
}



MultiStepRefiner::MultiStepRefiner() : width_(0), height_(0), img_left_(nullptr), cost_(nullptr),
cross_arms_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
min_disparity_(0), max_disparity_(0),
irv_ts_(0), irv_th_(0), lrcheck_thres_(LRCheckThres),
do_lr_check_(LRCheckOption), do_region_voting_(RegionVotingOption),
do_interpolating_(InterpolatingOption),
do_discontinuity_adjustment_(DiscontinuityAdjustmentOption) { }

MultiStepRefiner::~MultiStepRefiner()
{
}

bool MultiStepRefiner::Initialize(const sint32& width, const sint32& height)
{
	width_ = width;
	height_ = height;
	if (width_ <= 0 || height_ <= 0) {
		return false;
	}

	// 初始化边缘数据
	vec_edge_left_.clear();
	vec_edge_left_.resize(width * height);

	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));

	return true;
}

void MultiStepRefiner::SetData(const uint8* img_left, float32* cost, const CrossArm* cross_arms, float32* disp_left, float32* disp_right)
{
	img_left_ = img_left;
	cost_ = cost;
	cross_arms_ = cross_arms;
	disp_left_ = disp_left;
	disp_right_ = disp_right;
}

void MultiStepRefiner::SetParam(const sint32& min_disparity, const sint32& max_disparity, const sint32& irv_ts, const float32& irv_th, const float32& lrcheck_thres,
	const bool& do_lr_check, const bool& do_region_voting, const bool& do_interpolating, const bool& do_discontinuity_adjustment)
{
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	irv_ts_ = irv_ts;
	irv_th_ = irv_th;
	lrcheck_thres_ = lrcheck_thres;
	do_lr_check_ = do_lr_check;
	do_region_voting_ = do_region_voting;
	do_interpolating_ = do_interpolating;
	do_discontinuity_adjustment_ = do_discontinuity_adjustment;
}

void MultiStepRefiner::Refine()
{
	if (width_ <= 0 || height_ <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		cost_ == nullptr || cross_arms_ == nullptr) {
		return;
	}
	auto img_size = height_ * width_;
	// step1: outlier detection
	if (do_lr_check_) {
		//OutlierDetection();

		float32* disp_left_cuda = nullptr;
		float32* disp_right_cuda = nullptr;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_cuda, sizeof(float32) * img_size));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_right_cuda, sizeof(float32) * img_size));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_cuda, disp_left_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
		CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_right_cuda, disp_right_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		dim3 threadsPerBlock(32, 32);
		dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

		is_occlusions = (bool*)malloc(sizeof(bool) * img_size);
		is_mismatches = (bool*)malloc(sizeof(bool) * img_size);

		//uint8_t* img_left_cuda;
		//CUDA_CHECK_RETURN(cudaMalloc((void**)&img_left_cuda, sizeof(uint8_t) * img_size * 3));
		//CUDA_CHECK_RETURN(cudaMemcpyAsync(img_left_cuda, img_left_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));

		bool* is_occlusions_cuda = nullptr;
		bool* is_mismatches_cuda = nullptr;
		CUDA_CHECK_RETURN(cudaMalloc((void**)&is_occlusions_cuda, sizeof(bool) * img_size));
		CUDA_CHECK_RETURN(cudaMalloc((void**)&is_mismatches_cuda, sizeof(bool) * img_size));

		OutlierDetectionCuda << <blocksPerGrid, threadsPerBlock >> > (disp_left_cuda, disp_right_cuda,
			width_, height_, lrcheck_thres_, is_occlusions_cuda, is_mismatches_cuda);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_, disp_left_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost, stream1));//stream2?
		CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_right_, disp_right_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost, stream2));//stream2?

		CUDA_CHECK_RETURN(cudaMemcpyAsync(is_occlusions, is_occlusions_cuda, sizeof(bool) * img_size, cudaMemcpyDeviceToHost, stream1));
		CUDA_CHECK_RETURN(cudaMemcpyAsync(is_mismatches, is_mismatches_cuda, sizeof(bool) * img_size, cudaMemcpyDeviceToHost, stream2));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//cudaFree(img_left_cuda);
		cudaFree(is_occlusions_cuda);
		cudaFree(is_mismatches_cuda);
	}

	// step2: iterative region voting
	if (do_region_voting_) {
		//printf("正在IterativeRegionVoting\n");
		IterativeRegionVoting();
	}

	// step3: proper interpolation
	if (do_interpolating_) {
		//printf("正在ProperInterpolation\n");
		ProperInterpolation();
	}

	// step4: discontinuities adjustment
	if (do_discontinuity_adjustment_) {
		//printf("正在DepthDiscontinuityAdjustment\n");
		DepthDiscontinuityAdjustment();
	}
#if 1
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//中值滤波
	float32* disp_left_cuda = nullptr;
	float32* disp_left_dst_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_cuda, sizeof(float32) * img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_dst_cuda, sizeof(float32) * img_size));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_cuda, disp_left_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	// median filter
//adcensus_util::MedianFilter(disp_left_, disp_left_, width_, height_, 3);

	MedianFilter3x3 << <(img_size + MAX_DISPARITY - 1) / MAX_DISPARITY, MAX_DISPARITY >> >
		(disp_left_cuda, disp_left_dst_cuda, height_, width_);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_, disp_left_dst_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


#endif
}




__global__ void RegionVotingCuda(float32* disp_left, const uint32_t width, const uint32_t height, const sint32 max_disparity, const sint32 min_disparity,
	bool* outlier_marks, sint32	tau_s, float32 tau_h, const CrossArm* arms)
{

	const int y = blockIdx.x * blockDim.x + threadIdx.x;
	auto disp_range = max_disparity - min_disparity;

	if (y < height) {

		sint32* histogram = (sint32*)malloc(sizeof(sint32) * disp_range);
		if (histogram == nullptr) {
			printf("histogram NULL!!!!!!");
		}
		for (int x = 0; x < width; x++) {
			//判断是否为遮挡点或误匹配点
			if (outlier_marks[y * width + x]) {

				auto& disp = disp_left[y * width + x];
				if (disp == Invalid_Float) {

					//直方图置零
					memset(histogram, 0, disp_range * sizeof(sint32));

					// 计算支持区的视差直方图
					// 获取arm
					auto& arm = arms[y * width + x];
					// 遍历支持区像素视差，统计直方图
					for (sint32 t = -arm.top; t <= arm.bottom; t++) {
						const sint32& yt = y + t;
						auto& arm2 = arms[yt * width + x];
						for (sint32 s = -arm2.left; s <= arm2.right; s++) {
							const auto& d = disp_left[yt * width + x + s];
							if (d != Invalid_Float) {
								const auto di = lround(d);
								histogram[di - min_disparity]++;
							}
						}
					}
					// 计算直方图峰值对应的视差
					sint32 best_disp = 0, count = 0;
					sint32 max_ht = 0;
					for (sint32 d = 0; d < disp_range; d++) {
						const auto& h = histogram[d];
						if (max_ht < h) {
							max_ht = h;
							best_disp = d;
						}
						count += h;
					}

					if (max_ht > 0) {
						if (count > tau_s && max_ht * 1.0f / count > tau_h) {
							disp = best_disp + float(min_disparity);
						}
					}
				}
			}
		}
		free(histogram);
	}
}

__global__ void DeleteFilledPixel(bool* outlier_marks, float32* disp_left,
	const uint32_t width, const uint32_t height
)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	// 删除已填充像素
	if (y < height && x < width && disp_left[y * width + x] != Invalid_Float) {
		outlier_marks[y * width + x] = false;
		//printf("DeleteFilledPixel   ");
	}
}


void MultiStepRefiner::IterativeRegionVoting()
{
	const sint32 width = width_;

	//const auto disp_range = max_disparity_ - min_disparity_;


	auto img_size = width_ * height_;
	float32* disp_left_cuda = nullptr;
	//float32* disp_right_cuda = nullptr;
	CrossArm* cross_arms_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cross_arms_cuda, sizeof(CrossArm) * img_size));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cross_arms_cuda, cross_arms_, img_size * sizeof(CrossArm), cudaMemcpyHostToDevice, stream1));


	CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_cuda, sizeof(float32) * img_size));
	//CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_right_cuda, sizeof(float32) * img_size));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_cuda, disp_left_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_right_cuda, disp_right_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream2));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}


	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// 迭代5次
	const sint32 num_iters = IterativeNums;

	bool* is_occlusions_cuda = nullptr;
	bool* is_mismatches_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&is_occlusions_cuda, sizeof(bool) * img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&is_mismatches_cuda, sizeof(bool) * img_size));

	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_occlusions_cuda, is_occlusions, sizeof(bool) * img_size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_mismatches_cuda, is_mismatches, sizeof(bool) * img_size, cudaMemcpyHostToDevice, stream2));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	for (sint32 it = 0; it < num_iters; it++) {
		for (sint32 k = 0; k < 2; k++) {
			auto& outlier_marks_cuda = (k == 0) ? is_mismatches_cuda : is_occlusions_cuda;


			//printf("num_iters: %d\n\n\n", it);
			const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;


			RegionVotingCuda << <(height_ + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ >> > (disp_left_cuda,
				width_, height_, max_disparity_, min_disparity_,
				outlier_marks_cuda, irv_ts_, irv_th_, cross_arms_cuda);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());




			DeleteFilledPixel << <blocksPerGrid, threadsPerBlock >> >
				(outlier_marks_cuda, disp_left_cuda, width_, height_);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	}

	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_mismatches, is_mismatches_cuda, sizeof(bool) * img_size, cudaMemcpyDeviceToHost));
	for (int y = 0; y < height_; y++) {
		for (int x = 0; x < width_; x++) {

			if (is_mismatches[y * width + x]) {
				//printf("  is_mismatches  ");
			}
		}
	}
	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_occlusions, is_occlusions_cuda, sizeof(bool) * img_size, cudaMemcpyDeviceToHost));
	for (int y = 0; y < height_; y++) {
		for (int x = 0; x < width_; x++) {

			if (is_occlusions[y * width + x]) {
				//printf("  is_occlusions  ");
			}
		}
	}
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_, disp_left_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_right_, disp_right_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost));
	cudaFree(disp_left_cuda);
	//cudaFree(disp_right_cuda);
}
__global__ void FillDisp(const uint8* img_left, float32* disp_left,
	sint32 width, sint32 height, bool* is_mismatches, bool* is_occlusions, const sint32 max_search_length)
{
	// 遍历待处理像素
	const int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < height) {
		for (int x = 0; x < width; x++) {
			if ((is_occlusions[y * width + x] || is_mismatches[y * width + x]))
			{
				//存储16个方向的最近有效视差值
				float32* disp_collects_d = (float32*)malloc(sizeof(float32) * 16);//存储视差值
				sint32* disp_collects_idx = (sint32*)malloc(sizeof(sint32) * 16);//存储索引
				uint8 disp_collects_num = 0;//已存储的个数

				if (!disp_collects_d || !disp_collects_idx) {
					printf("null!!!!");
				}

				double ang = 0.0;
				for (sint32 s = 0; s < 16; s++) {
					const auto sina = sin(ang);
					const auto cosa = cos(ang);
					//分别沿16个方向搜索
					for (sint32 m = 1; m < max_search_length; m++) {

						const sint32 yy = lround(y + m * sina);
						const sint32 xx = lround(x + m * cosa);
						if (yy < 0 || yy >= height || xx < 0 || xx >= width) { break; }

						const auto& d = disp_left[yy * width + xx];
						if (d != Invalid_Float) {

							disp_collects_d[disp_collects_num] = d;
							disp_collects_idx[disp_collects_num] = yy * width * 3 + 3 * xx;
							disp_collects_num++;

							break;
						}
					}

					ang += PI / 16;
				}
				//无有效视差
				if (!disp_collects_num) {
					return;
				}


				if (is_mismatches[y * width + x]) {
					sint32 min_color_diff = 9999;
					float32 d = 0.0f;
					//当前的的RGB
					const auto color_r = img_left[y * width * 3 + 3 * x];
					const auto color_g = img_left[y * width * 3 + 3 * x + 1];
					const auto color_b = img_left[y * width * 3 + 3 * x + 2];


					for (int i = 0; i < disp_collects_num; i++) {
						//存储点的RGB
						const auto color2_r = img_left[disp_collects_idx[i]];
						const auto color2_g = img_left[disp_collects_idx[i] + 1];
						const auto color2_b = img_left[disp_collects_idx[i] + 2];
						const auto color_diff = abs(color_r - color2_r)
							+ abs(color_g - color2_g) + abs(color_b - color2_b);
						if (min_color_diff > color_diff) {
							min_color_diff = color_diff;
							d = disp_collects_d[i];
						}
					}
					//用最小色差的点填充
					disp_left[y * width + x] = d;
				}
				else {
					float32 min_disp = Large_Float;
					for (int i = 0; i < disp_collects_num; i++) {

						min_disp = MIN(disp_collects_d[i], min_disp);
					}
					//最小视差值填充
					disp_left[y * width + x] = min_disp;
				}

				free(disp_collects_d);
				free(disp_collects_idx);

			}
		}

	}

}

void MultiStepRefiner::ProperInterpolation()
{

	auto img_size = width_ * height_;

	//const float32 pi = 3.1415926f;
	// 最大搜索行程，没有必要搜索过远的像素
	const sint32 max_search_length = abs(max_disparity_) * MaxSearchLengthTimes;

	float32* disp_left_cuda = nullptr;
	uint8* img_left_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_cuda, sizeof(float32) * img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_left_cuda, sizeof(uint8) * img_size * 3));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_cuda, disp_left_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_left_cuda, img_left_, sizeof(uint8) * img_size * 3, cudaMemcpyHostToDevice, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);

	bool* is_occlusions_cuda = nullptr;
	bool* is_mismatches_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&is_occlusions_cuda, sizeof(bool) * img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&is_mismatches_cuda, sizeof(bool) * img_size));

	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_occlusions_cuda, is_occlusions, sizeof(bool) * img_size, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(is_mismatches_cuda, is_mismatches, sizeof(bool) * img_size, cudaMemcpyHostToDevice, stream2));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;

	FillDisp << <(height_ + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ >> > (img_left_cuda, disp_left_cuda,
		width_, height_, is_mismatches_cuda, is_occlusions_cuda, max_search_length);
	//FillDisp << <(height_ + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ >> > (img_left_cuda, disp_left_cuda,
	//	width_, height_, is_mismatches_cuda, is_occlusions_cuda, max_search_length);

	//FillDisp << <blocksPerGrid, threadsPerBlock >> > (img_left_cuda, disp_left_cuda,
	//	width_, height_, is_mismatches_cuda, is_occlusions_cuda, max_search_length);
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_, disp_left_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost, stream2));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaFree(disp_left_cuda);
	cudaFree(img_left_cuda);
	cudaFree(is_occlusions_cuda);
	cudaFree(is_mismatches_cuda);
}

__global__ void EdgeDispOptmize(float32* disp_left, float32* cost, uint8* edge_mask,
	const uint32_t width, const uint32_t height, sint32 max_disparity, sint32 min_disparity)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto disp_range = max_disparity - min_disparity;
	if (y < height && x < width - 1 && x >= 1) {

		const auto& e_label = edge_mask[y * width + x];
		if (e_label == 1) {
			const auto disp_ptr = disp_left + y * width;
			float32& d = disp_ptr[x];
			if (d != Invalid_Float) {
				const sint32& di = lround(d);
				const auto cost_ptr = cost + y * width * disp_range + x * disp_range;
				float32 c0 = cost_ptr[di];

				// 记录左右两边像素的视差值和代价值
				// 选择代价最小的像素视差值
				//for (int k = 0; k < 2; k++) {
				//	const sint32 x2 = (k == 0) ? x - 1 : x + 1;
				//	const float32& d2 = disp_ptr[x2];
				//	const sint32& d2i = lround(d2);
				//	if (d2 != Invalid_Float) {
				//		const auto& c = (k == 0) ? cost_ptr[-disp_range + d2i] : cost_ptr[disp_range + d2i];
				//		if (c < c0) {
				//			d = d2;
				//			c0 = c;
				//		}
				//	}
				//}

				//const sint32 x_l = x - 1 ,x_r= x + 1;
				//const float32& disp_l= disp_ptr[x_l];
				//const float32& disp_r= disp_ptr[x_r];
				//if (disp_l != Invalid_Float && disp_r != Invalid_Float) {
				//	c0=MIN(cost_ptr[-disp_range + lround(disp_l)], 
				//		cost_ptr[disp_range + lround(disp_r)]);
				//}

				for (int k = 0; k < 2; k++) {
					const sint32 x2 = (k == 0) ? x - 1 : x + 1;
					const float32& d2 = disp_ptr[x2];
					const sint32& d2i = lround(d2);
					if (d2 != Invalid_Float) {
						const auto& c = (k == 0) ? cost_ptr[-disp_range + d2i] : cost_ptr[disp_range + d2i];
						if (c < c0) {
							d = d2;
							c0 = c;
						}
					}
				}







			}
		}

	}


}

void MultiStepRefiner::DepthDiscontinuityAdjustment()
{
	const sint32 width = width_;
	const sint32 height = height_;
	const auto disp_range = max_disparity_ - min_disparity_;
	if (disp_range <= 0) {
		return;
	}

	// 对视差图做边缘检测
	// 边缘检测的方法是灵活的，这里选择sobel算子
	const float32 edge_thres = 5.0f;

	const auto img_size = width * height;

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (width_ + block_size.x - 1) / block_size.x;
	grid_size.y = (height_ + block_size.y - 1) / block_size.y;

	uint8* edge_mask = &vec_edge_left_[0];
	memset(edge_mask, 0, width * height * sizeof(uint8));
	uint8* edge_mask_cuda = nullptr;
	float32* disp_left_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&edge_mask_cuda, width * height * sizeof(uint8)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&disp_left_cuda, sizeof(float32) * img_size));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_cuda, disp_left_, sizeof(float32) * img_size, cudaMemcpyHostToDevice, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	EdgeDetectCuda << <grid_size, block_size, 0, stream1 >> > (edge_mask_cuda, disp_left_cuda, width, height, edge_thres);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpyAsync(edge_mask, edge_mask_cuda, width * height * sizeof(uint8), cudaMemcpyDeviceToHost, stream1));//stream2?
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	float32* cost_cuda = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_cuda, width * height * disp_range * sizeof(float32)));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_cuda, cost_, width * height * disp_range * sizeof(float32), cudaMemcpyHostToDevice, stream1));//stream2?


																																		 // 调整边缘像素的视差
	EdgeDispOptmize << <grid_size, block_size, 0, stream1 >> > (disp_left_cuda, cost_cuda, edge_mask_cuda,
		width_, height_, max_disparity_, min_disparity_);
	CUDA_CHECK_RETURN(cudaMemcpyAsync(disp_left_, disp_left_cuda, sizeof(float32) * img_size, cudaMemcpyDeviceToHost, stream2));

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	//free();
	cudaFree(disp_left_cuda);
	cudaFree(cost_cuda);
	cudaFree(edge_mask_cuda);
}

