#include "scanline_optimizer.h"


__global__ void ScanlineOptimizeLeftRight(const uint8* img_left, const uint8* img_right,
	const float32* cost_so_src, float32* cost_so_dst, bool is_forward,
	const sint32 width, const sint32 height, const sint32  min_disparity,
	const sint32 max_disparity, const float32  p1, const float32  p2, const sint32  tso)
{

	const int idy = blockIdx.x * blockDim.x + threadIdx.x;


	const sint32 direction = is_forward ? 1 : -1;
	const sint32 disp_range = MaxDisparity;
	
	////return;
	////__syncthreads();
	//return;

	if (idy < height)
	{
		
		//return;

		// 路径头为每一行的首(尾,dir=-1)列像素

		auto cost_init_row = (is_forward) ? (cost_so_src + idy * width * disp_range) : (cost_so_src + idy * width * disp_range + (width - 1) * disp_range);
		auto cost_aggr_row = (is_forward) ? (cost_so_dst + idy * width * disp_range) : (cost_so_dst + idy * width * disp_range + (width - 1) * disp_range);
		auto img_row = (is_forward) ? (img_left + idy * width * 3) : (img_left + idy * width * 3 + 3 * (width - 1));
		
		const auto img_row_r = img_right + idy * width * 3;
		sint32 x = (is_forward) ? 0 : width - 1;

		// 路径上当前颜色值和上一个颜色值
		uint8 color_r = img_row[0], color_g = img_row[1], color_b = img_row[2];
		uint8 color_last_r = color_r, color_last_g = color_g, color_last_b = color_b;
		//ADColor color_last = color;
		
		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）

		
		float32* cost_last_path = (float32*)malloc(sizeof(float32) * disp_range + 2);
		//arrs[Idx] = (T*)malloc(sizeof(T) * size);
		//memset(arrs[Idx], 0, sizeof(T) * size);
		memset(cost_last_path, Large_Float, disp_range + 2);
		//return;
		// 初始化：第一个像素的聚合代价值等于初始代价值
		
		/*CUDA_CHECK_RETURN(cudaMemcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(float32), cudaMemcpyDeviceToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(float32), cudaMemcpyDeviceToDevice));*/
		memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(float32));
		memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(float32));
		cost_init_row += direction * disp_range;
		cost_aggr_row += direction * disp_range;
		img_row += direction * 3;
		x += direction;
		
		//return;
		//__syncthreads();
		// 路径上上个像素的最小代价值
		float32 mincost_last_path = Large_Float;
		for (int i = 0; i < disp_range + 2; i++) {
			mincost_last_path = MIN(mincost_last_path, cost_last_path[i]);
		}
		

		// 自方向上第2个像素开始按顺序聚合
		for (sint32 j = 0; j < width - 1; j++)
		{

			color_r = img_row[0], color_g = img_row[1], color_b = img_row[2];

			const uint8 d1 = MAX(abs(color_r - color_last_r),
				MAX(abs(color_g - color_last_g), abs(color_b - color_last_b)));
			uint8 d2 = d1;
			float32 min_cost = Large_Float;
			for (sint32 d = 0; d < disp_range; d++) {
				return;
				const sint32 xr = x - d- min_disparity;
				if (xr > 0 && xr < width - 1) {
					
					const uint8 color_rr = img_row_r[3 * xr],
						color_rg = img_row_r[3 * xr + 1], color_rb = img_row_r[3 * xr + 2];
					//const ADColor color_r = ADColor(img_row_r[3 * xr], img_row_r[3 * xr + 1], img_row_r[3 * xr + 2]);
					const uint8 color_last_rr = img_row_r[3 * (xr - direction)],
						color_last_rg = img_row_r[3 * (xr - direction) + 1],
						color_last_rb = img_row_r[3 * (xr - direction) + 2];


					d2 = MAX(abs(color_rr - color_last_rr),
						MAX(abs(color_rg - color_last_rg), abs(color_rb - color_last_rb)));

				}
				

				// 计算P1和P2
				float32 P1(0.0f), P2(0.0f);
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 >= tso) {
					P1 = p1 / 10; P2 = p2 / 10;
				}

				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const float32  cost = cost_init_row[d];
				const float32 l1 = cost_last_path[d + 1];
				const float32 l2 = cost_last_path[d] + P1;
				const float32 l3 = cost_last_path[d + 2] + P1;
				const float32 l4 = mincost_last_path + P2;

				float32 cost_s = cost + static_cast<float32>(MIN(MIN(l1, l2), MIN(l3, l4)));
				cost_s /= 2;

				cost_aggr_row[d] = cost_s;
				min_cost = MIN(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			//memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(float32));
			memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(float32));



			// 下一个像素
			cost_init_row += direction * disp_range;
			cost_aggr_row += direction * disp_range;
			img_row += direction * 3;
			x += direction;

			// 像素值重新赋值
			color_last_r = color_r, color_last_g = color_g, color_last_b = color_b;
		}

		free(cost_last_path);

	}
else {
//return;
}

//return;
}



__global__ void ScanlineOptimizeUpDown(const uint8* img_left, const uint8* img_right,
	const float32* cost_so_src, float32* cost_so_dst, bool is_forward,
	const sint32 width, const sint32 height, const sint32  min_disparity,
	const sint32 max_disparity, const float32  p1, const float32  p2, const sint32  tso,const uint32 num)

{



	// 视差范围
	const sint32 disp_range = max_disparity - min_disparity;

	const sint32 direction = is_forward ? 1 : -1;

	// 聚合

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	auto left_width = num  * width / 2;
	auto right_width= (num+1)  * width / 2;

	if (idx < right_width && idx >= left_width)
	{
		// 路径头为每一列的首(尾,dir=-1)行像素
		auto cost_init_col = (is_forward) ? (cost_so_src + idx * disp_range) : (cost_so_src + (height - 1) * width * disp_range + idx * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_so_dst + idx * disp_range) : (cost_so_dst + (height - 1) * width * disp_range + idx * disp_range);
		auto img_col = (is_forward) ? (img_left + 3 * idx) : (img_left + (height - 1) * width * 3 + 3 * idx);
		sint32 y = (is_forward) ? 0 : height - 1;

		// 当前color与前一个color
		uint8 color_r = img_col[0], color_g = img_col[1], color_b = img_col[2];
		uint8 color_last_r = color_r, color_last_g = color_g, color_last_b = color_b;



		//存储上一个像素的cost
		float32* last_cost = (float32*)malloc(sizeof(float32) * disp_range + 2);
		if (last_cost ==nullptr) {
			printf("ScanlineOptimizeUpDown NULL !!!  %u\n", right_width);
		}
		memset(last_cost, Large_Float, disp_range + 2);

		// initial
		//第一个像素对应自身cost

		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(float32));
		memcpy(&last_cost[1], cost_aggr_col, disp_range * sizeof(float32));
		cost_init_col += direction * width * disp_range;
		cost_aggr_col += direction * width * disp_range;
		img_col += direction * width * 3;
		y += direction;

		// 前一像素的最小cost

		float32 last_min_cost = Large_Float;
		for (int i = 0; i < disp_range + 2; i++) {
			last_min_cost = MIN(last_min_cost, last_cost[i]);
		}


		// 自方向上第2个像素开始按顺序聚合
		for (sint32 i = 0; i < height - 1; i++) {

			color_r = img_col[0], color_g = img_col[1], color_b = img_col[2];

			//左视图相邻像素p和p-r颜色差 d1
			const uint8 d1 = MAX(abs(color_r - color_last_r),
				MAX(abs(color_g - color_last_g), abs(color_b - color_last_b)));
			uint8 d2 = d1;
			float32 min_cost = Large_Float;


			
			//右视图对应同名点相邻像素pd和 pd-r 颜色差
			for (sint32 d = 0; d < disp_range; d++) {
				const sint32 xr = idx - d - min_disparity;
				if (xr > 0 && xr < width - 1) {

					const uint8 color_rr = img_right[y * width * 3 + 3 * xr],
						color_rg = img_right[y * width * 3 + 3 * xr + 1], color_rb=img_right[y * width * 3 + 3 * xr + 2];
				
					const uint8 color_last_rr = img_right[(y - direction) * width * 3 + 3 * xr],
						color_last_rg = img_right[(y - direction) * width * 3 + 3 * xr + 1],
						color_last_rb = img_right[(y - direction) * width * 3 + 3 * xr + 2];

					d2 = MAX(abs(color_rr - color_last_rr),
						MAX(abs(color_rg - color_last_rg), abs(color_rb - color_last_rb)));

				}
				
				
				// P1和P2 所需阈值按论文给出
				float32 P1(0.0f), P2(0.0f);
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 >= tso) {
					P1 = p1 / 10; P2 = p2 / 10;
				}

			
				const float32  cost = cost_init_col[d];
				const float32 l1 = last_cost[d + 1];
				const float32 l2 = last_cost[d] + P1;
				const float32 l3 = last_cost[d + 2] + P1;
				const float32 l4 = last_min_cost + P2;

				float32 cost_s = cost + static_cast<float32>(MIN(MIN(l1, l2), MIN(l3, l4)));
				cost_s /= 2;

				cost_aggr_col[d] = cost_s;
				min_cost = MIN(min_cost, cost_s);

			}

			// 重置上个像素的最小代价值和代价数组
			last_min_cost = min_cost;
			memcpy(&last_cost[1], cost_aggr_col, disp_range * sizeof(float32));

			// 下一个像素
			cost_init_col += direction * width * disp_range;
			cost_aggr_col += direction * width * disp_range;
			img_col += direction * width * 3;
			y += direction;

			// 更新color
			color_last_r = color_r, color_last_g = color_g, color_last_b = color_b;
		}
		free(last_cost);
	}
}



ScanlineOptimizer::ScanlineOptimizer() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
cost_init_(nullptr), cost_aggr_(nullptr),
min_disparity_(0), max_disparity_(0),
so_p1_(0), so_p2_(0),
so_tso_(0) {}

ScanlineOptimizer::~ScanlineOptimizer() {}

void ScanlineOptimizer::SetData(const uint8* img_left, const uint8* img_right, float32* cost_init,
	float32* cost_aggr)
{
	img_left_ = img_left;
	img_right_ = img_right;
	//CUDA_CHECK_RETURN(cudaMallocManaged((void**)&census_left_, sizeof(uint64) * img_size));
	//CUDA_CHECK_RETURN(cudaMallocManaged((void**)&census_right_, sizeof(uint64) * img_size));
	cost_init_ = cost_init;
	cost_aggr_ = cost_aggr;
}

void ScanlineOptimizer::SetParam(const sint32& width, const sint32& height, const sint32& min_disparity,
	const sint32& max_disparity, const float32& p1, const float32& p2, const sint32& tso)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	so_p1_ = p1;
	so_p2_ = p2;
	so_tso_ = tso;
}

void ScanlineOptimizer::Optimize()
{
	if (width_ <= 0 || height_ <= 0 ||
		img_left_ == nullptr || img_right_ == nullptr ||
		cost_init_ == nullptr || cost_aggr_ == nullptr) {
		return;
	}
	static cudaStream_t stream1, stream2, stream3;

	CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));


	// 4方向扫描线优化
	// 模块的首次输入是上一步代价聚合后的数据，也就是cost_aggr_
	// 我们把四个方向的优化按次序进行，并利用cost_init_及cost_aggr_间次保存临时数据，这样不用开辟额外的内存来存储中间结果
	// 模块的最终输出也是cost_aggr_


	// Cost Aggregation
	//const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE / 64;
	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE / WARP_SIZE;
	//const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / 64;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;

	uint8_t* img_left_cuda;
	uint8_t* img_right_cuda;
	float32* cost_aggr_cuda;
	float32* cost_init_cuda;
	const sint32 img_size = width_ * height_;

	//sint32 *width = nullptr;
	//sint32 *height = nullptr;
	//CUDA_CHECK_RETURN(cudaMallocManaged((void**) &width, sizeof(sint32)));
	//CUDA_CHECK_RETURN(cudaMallocManaged((void**) &height, sizeof(sint32)));
	//*width = width_;
	//*height = height_;

	//memcpy(&cost_aggr_[0], cost_init_, width_ * height_ * MAX_DISPARITY * sizeof(float32));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_aggr_cuda, width_ * height_ * MAX_DISPARITY * sizeof(float32)));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_aggr_cuda, cost_aggr_, width_ * height_ * MAX_DISPARITY * sizeof(float32), cudaMemcpyHostToDevice, stream3));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_init_cuda, width_ * height_ * MAX_DISPARITY * sizeof(float32)));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_init_cuda, cost_init_, width_ * height_ * MAX_DISPARITY * sizeof(float32), cudaMemcpyHostToDevice, stream2));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_left_cuda, sizeof(uint8_t) * img_size * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&img_right_cuda, sizeof(uint8_t) * img_size * 3));

	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_left_cuda, img_left_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(img_right_cuda, img_right_, sizeof(uint8_t) * img_size * 3, cudaMemcpyHostToDevice, stream3));//stream2?

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	// left to right
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
#if ScanlineOption
	ScanlineOptimizeLeftRight << <(height_ + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream1 >> >
		(img_left_cuda, img_right_cuda, cost_aggr_cuda, cost_init_cuda, true,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_);


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	///CostAggregationKernelLeftToRight << <(rows + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2 >> > (d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}


	 //right to left
	ScanlineOptimizeLeftRight << <(height_ + PIXELS_PER_BLOCK_HORIZ - 1) / PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3 >> >
		(img_left_cuda, img_right_cuda, cost_init_cuda, cost_aggr_cuda, false,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s %d\n", cudaGetErrorString(err), err);
		exit(-1);
	}



	//上下扫描 大图时显存不足 因此分左右两块分别做扫描线优化

	 //up to down

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	ScanlineOptimizeUpDown << <(width_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream3 >> >
		(img_left_cuda, img_right_cuda, cost_aggr_cuda , cost_init_cuda, true,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_,0);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	ScanlineOptimizeUpDown << <(width_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream3 >> >
		(img_left_cuda, img_right_cuda, cost_aggr_cuda, cost_init_cuda, true,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_, 1);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());


	// down to up
	ScanlineOptimizeUpDown << <(width_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream2 >> >
		(img_left_cuda, img_right_cuda, cost_init_cuda , cost_aggr_cuda, false,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_,0);


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	ScanlineOptimizeUpDown << <(width_ + PIXELS_PER_BLOCK - 1) / PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream2 >> >
		(img_left_cuda, img_right_cuda, cost_init_cuda, cost_aggr_cuda, false,
			width_, height_, min_disparity_,
			max_disparity_, so_p1_, so_p2_, so_tso_,1);


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_aggr_, cost_aggr_cuda, width_ * height_ * MAX_DISPARITY * sizeof(float32), cudaMemcpyDeviceToHost, stream3));
	
	//CUDA_CHECK_RETURN(cudaMalloc((void**)&cost_init_cuda, width_ * height_ * MAX_DISPARITY * sizeof(float32)));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(cost_init_, cost_init_cuda, width_ * height_ * MAX_DISPARITY * sizeof(float32), cudaMemcpyDeviceToHost, stream2));
#endif	
	// Cuda Free
	cudaFree(cost_aggr_cuda);
	cudaFree(cost_init_cuda);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//cudaEventRecord(stop, 0);
	//float elapsed_time_ms;
	//cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//std::cout << "scanline optimizing cuda:" << elapsed_time_ms << "ms" << std::endl;

}

