#include"sgm_utils.h"
#include<cstdio>
#include<vector>
#include <iostream>

//5x5是因为uint32的大小限制
void census_transform_5x5(const uint8* grayscale_img, uint32* census, const sint32& width, const sint32& height)
{
	if (grayscale_img == nullptr || census == nullptr || width <= 5u || height <= 5u) {
		return;
	}
	uint8 grayscale_center = 0;
	uint8 grayscale_neighbour = 0;
	// 逐像素计算census值
	// 最头和最尾的cost都为0
//#pragma omp parallel for 无显著提升
	for (sint32 i = 2; i < height - 2; i++) {
		for (sint32 j = 2; j < width - 2; j++) {
			// 灰度图像中心像素值
			grayscale_center = grayscale_img[i * width + j];
			// 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
			uint32 census_bitstream = 0u;
			for (sint32 r = -2; r <= 2; r++) {
				for (sint32 c = -2; c <= 2; c++) {
					//向左移一位，相当于在二进制后面增添0，相等于*2
					census_bitstream <<= 1;
					grayscale_neighbour = grayscale_img[(i + r) * width + j + c];
					if (grayscale_neighbour < grayscale_center) {
						census_bitstream += 1;
					}
				}
			}
			// 中心像素的census值
			// 这里就是一个25bit的比特串，用于后面对其进行汉明距离的计算
			census[i * width + j] = census_bitstream;
		}
	}
}

//汉明距离计算
//if your compiler supports 64-bit integers 直接使用硬件寄存器加速
uint8 hamming_distance(const uint32 x, const uint32 y)
{
	//数1的个数
	return __builtin_popcountll(x ^ y);
}

void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, 
	const sint32& min_disparity, const sint32& max_disparity, 
	const sint32& p1, const sint32& p2_init, 
	const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
	if (!(width > 0 && height > 0 && max_disparity > min_disparity)){
		std::cout << "image/parameter error" << std::endl;
		return;
	}

	// 视差范围
	const sint32 disparity_range = max_disparity - min_disparity;


	// 正向(左->右) ：is_forward = true ; direction = 1
	// 反向(右->左) ：is_forward = false; direction = -1;
	sint32 direction;
	if (is_forward){
		direction = 1;
	}
	else {
		direction = -1;
	}

	// 聚合
#pragma omp parallel for
	// i =0是第一行
	for (sint32 i = 0u; i < height; i++) {
		//左右路径聚合，按照每一行来考虑，根据方向决定初始代价值为第一列还是最后一列
		// 每条聚合路径的开头，只放一个像素点的内容
		uint8* cost_init_of_row;
		uint8* cost_aggr_of_row;
		uint8* current_gray_row;
		// 初始化赋值路径头
		if (is_forward) {
			// 指向第i+1行的第一个像素点灰度值
			current_gray_row = (uint8*)img_data + i * width;
			// 初始的代价取的都是第一个视差的
			cost_init_of_row = (uint8*) cost_init + i * width * disparity_range;
			cost_aggr_of_row = cost_aggr + i * width * disparity_range;
		}
		else {
			// 指向第i+2行第一个像素点的前面一个像素点 = 第i+1行最后一个
			current_gray_row = (uint8*)img_data + (i + 1) * width - 1;
			cost_init_of_row = (uint8*)cost_init + ((i + 1) * width - 1) * disparity_range;
			cost_aggr_of_row = cost_aggr + i * width * disparity_range + (width - 1) * disparity_range;
		}

		// 当前行第一个像素点的灰度值和上一个像素点灰度值的初始化（都赋值为该行的第一个像素点）
		uint8 gray = *current_gray_row;
		uint8 gray_last = *current_gray_row;

		// 路径上上个像素的代价数组，存放视差数量个元素，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8> cost_last_pixel(disparity_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值，即忽略第一个像素点的聚合
		memcpy(cost_aggr_of_row, cost_init_of_row, disparity_range * sizeof(uint8));
		// 防止溢出，访问下标从1开始
		memcpy(&cost_last_pixel[1], cost_aggr_of_row, disparity_range * sizeof(uint8));
		// 移动到每条路径的第二个像素点，开始迭代代价聚合
		cost_init_of_row += direction * disparity_range;
		cost_aggr_of_row += direction * disparity_range;
		current_gray_row += direction;

		// 路径上上个像素的最小代价值
		uint8 mincost_last_pixel = UINT8_MAX;
		for (auto cost : cost_last_pixel) {
			mincost_last_pixel = std::min(mincost_last_pixel, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (sint32 j = 0; j < width - 1; j++) {
			gray = *current_gray_row;
			uint8 min_cost = UINT8_MAX;
			for (sint32 d = 0; d < disparity_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				// C(p,d)
				const uint8  cost = cost_init_of_row[d];
				// L(p−r,d)表示路径内上一个像素视差为d时的聚合代价值
				const uint16 l1 = cost_last_pixel[d + 1]; // 从1开始，所以这里加1，表示的仍然是上一个像素
				// L(p−r,d−1) + P1, 表示路径内上一个像素视差为d-1时的聚合代价值
				const uint16 l2 = cost_last_pixel[d] + p1;
				// L(p−r,d+1)+ P1, 表示路径内上一个像素视差为d+1时的聚合代价值
				const uint16 l3 = cost_last_pixel[d + 2] + p1;
				// mincost_last_pixel = min(L(p−r,i))，表示路径内上一个像素所有代价值的最小值
				const uint16 l4 = mincost_last_pixel + std::max(p1, p2_init / (abs(gray - gray_last) + 1));// 分母+1,防止分母为0
				// 计算结果为cost_s
				const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_pixel);
				cost_aggr_of_row[d] = cost_s;
				// 更新该行最小代价，省去再遍历一次的麻烦
				min_cost = std::min(min_cost, cost_s);
			}

			// 更新上个像素的最小代价值和代价数组
			mincost_last_pixel = min_cost;
			memcpy(&cost_last_pixel[1], cost_aggr_of_row, disparity_range * sizeof(uint8));

			// 下一个像素
			cost_init_of_row += direction * disparity_range;
			cost_aggr_of_row += direction * disparity_range;
			current_gray_row += direction;

			// 灰度像素值重新赋值
			gray_last = gray;
		}
	}
}

void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity, 
	const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
	if (!(width > 0 && height > 0 && max_disparity > min_disparity)) {
		std::cout << "image/parameter error" << std::endl;
		return;
	}

	// 视差范围
	const sint32 disparity_range = max_disparity - min_disparity;

	// 正向(上->下) ：is_forward = true ; direction = 1
	// 反向(下->上) ：is_forward = false; direction = -1;
	const sint32 direction = is_forward ? 1 : -1;

	// 聚合
#pragma omp parallel for
	for (sint32 j = 0; j < width; j++) {
		uint8* cost_init_of_col;
		uint8* cost_aggr_of_col;
		uint8* current_gray_col;
		// 初始化赋值路径头
		if (is_forward) {
			// 指向第i+1列的第一个像素点灰度值
			current_gray_col = (uint8*)img_data + j;
			// 初始的代价取的都是第一个视差的
			cost_init_of_col = (uint8*)cost_init + j * disparity_range;
			cost_aggr_of_col = cost_aggr + j * disparity_range;
		}
		else {
			// 指向第i+2列第一个像素点的前面一个像素点 = 第i+1列最后一个
			current_gray_col = (uint8*)img_data + (height - 1) * width + j;
			cost_init_of_col = (uint8*)cost_init + (height - 1) * width * disparity_range + j * disparity_range;
			cost_aggr_of_col = cost_aggr + (height - 1) * width * disparity_range + j * disparity_range;
		}
		// 路径上当前灰度值和上一个灰度值
		uint8 gray = *current_gray_col;
		uint8 gray_last = *current_gray_col;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8> cost_last_pixel(disparity_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_of_col, cost_init_of_col, disparity_range * sizeof(uint8));
		memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));
		cost_init_of_col += direction * width * disparity_range;
		cost_aggr_of_col += direction * width * disparity_range;
		current_gray_col += direction * width;

		// 路径上上个像素的最小代价值
		uint8 mincost_last_pixel = UINT8_MAX;
		for (auto cost : cost_last_pixel) {
			mincost_last_pixel = std::min(mincost_last_pixel, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (sint32 i = 0; i < height - 1; i++) {
			gray = *current_gray_col;
			uint8 min_cost = UINT8_MAX;
			for (sint32 d = 0; d < disparity_range; d++) {
				const uint8  cost = cost_init_of_col[d];
				const uint16 l1 = cost_last_pixel[d + 1];
				const uint16 l2 = cost_last_pixel[d] + p1;
				const uint16 l3 = cost_last_pixel[d + 2] + p1;
				const uint16 l4 = mincost_last_pixel + std::max(p1, p2_init / (abs(gray - gray_last) + 1));
				const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_pixel);

				cost_aggr_of_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}
			mincost_last_pixel = min_cost;
			memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));
			cost_init_of_col += direction * width * disparity_range;
			cost_aggr_of_col += direction * width * disparity_range;
			current_gray_col += direction * width;

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void CostAggregateDagonal_1(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity, const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward)
	// 左上到右下偏移量是(w + 1) 个像素((w + 1) * disp_range个代价)
	// 对角线路径中的左上到右下，位置偏移就是行号列号各加1
{
	if (!(width > 1 && height > 1 && max_disparity > min_disparity)) {
		std::cout << "image/parameter error" << std::endl;
		return;
	}

	// 视差范围
	const sint32 disparity_range = max_disparity - min_disparity;

	// 正向(左上->右下) ：is_forward = true ; direction = 1
	// 反向(右下->左上) ：is_forward = false; direction = -1;
	const sint32 direction = is_forward ? 1 : -1;

	// 存储当前的行列号，判断是否到达影像边界
	sint32 current_row = 0;
	sint32 current_col = 0;
//#pragma omp parallel for
	for (sint32 j = 0; j < width; j++) {
		// 路径头为每一列的首(尾,dir=-1)行像素
		uint8* cost_init_of_col;
		uint8* cost_aggr_of_col;
		uint8* current_gray_col;
		// 初始化赋值路径头
		if (is_forward) {
			// 指向第i+1行的第一个像素点灰度值
			current_gray_col = (uint8*)img_data + j;
			// 初始的代价取的都是第一个视差的
			cost_init_of_col = (uint8*)cost_init + j * disparity_range;
			cost_aggr_of_col = cost_aggr + j * disparity_range;
		}
		else {
			// 指向第i+2行第一个像素点的前面一个像素点 = 第i+1行最后一个
			current_gray_col = (uint8*)img_data + (height - 1) * width + j;
			cost_init_of_col = (uint8*)cost_init + (height - 1) * width * disparity_range + j * disparity_range;
			cost_aggr_of_col = cost_aggr + (height - 1) * width * disparity_range + j * disparity_range;
		}
		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8> cost_last_pixel(disparity_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_of_col, cost_init_of_col, disparity_range * sizeof(uint8));
		memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));

		// 路径上当前灰度值和上一个灰度值
		uint8 gray = *current_gray_col;
		uint8 gray_last = *current_gray_col;

		// 对角线路径上的下一个像素，中间间隔width+1个像素
		// 这里要多一个边界处理
		// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == width - 1 && current_row < height - 1) {
			// 左上->右下，碰右边界
			cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range;
			cost_aggr_of_col = (uint8*)cost_aggr + (current_row + direction) * width * disparity_range;
			current_gray_col = (uint8*)img_data + (current_row + direction) * width;
			current_col = 0;
		}
		else if (!is_forward && current_col == 0 && current_row > 0) {
			// 右下->左上，碰左边界
			cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
			cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
			current_gray_col = (uint8*)img_data + (current_row + direction) * width + (width - 1);
			current_col = width - 1;
		}
		else {
			cost_init_of_col += direction * (width + 1) * disparity_range;
			cost_aggr_of_col += direction * (width + 1) * disparity_range;
			current_gray_col += direction * (width + 1);
		}

		// 路径上上个像素的最小代价值
		uint8 mincost_last_pixel = UINT8_MAX;
		for (auto cost : cost_last_pixel) {
			mincost_last_pixel = std::min(mincost_last_pixel, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (sint32 i = 0; i < height - 1; i++) {
			gray = *current_gray_col;
			uint8 min_cost = UINT8_MAX;
			for (sint32 d = 0; d < disparity_range; d++) {
				const uint8  cost = cost_init_of_col[d];
				const uint16 l1 = cost_last_pixel[d + 1];
				const uint16 l2 = cost_last_pixel[d] + p1;
				const uint16 l3 = cost_last_pixel[d + 2] + p1;
				const uint16 l4 = mincost_last_pixel + std::max(p1, p2_init / (abs(gray - gray_last) + 1));
				const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_pixel);
				cost_aggr_of_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_pixel = min_cost;
			memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));

			// 当前像素的行列号
			current_row += direction;
			current_col += direction;

			// 下一个像素,这里要多一个边界处理
			// 这里要多一个边界处理
			// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
			if (is_forward && current_col == width - 1 && current_row < height - 1) {
				// 左上->右下，碰右边界
				cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range;
				cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range;
				current_gray_col = (uint8*)img_data + (current_row + direction) * width;
				current_col = 0;
			}
			else if (!is_forward && current_col == 0 && current_row > 0) {
				// 右下->左上，碰左边界
				cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
				cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
				current_gray_col = (uint8*)img_data + (current_row + direction) * width + (width - 1);
				current_col = width - 1;
			}
			else {
				cost_init_of_col += direction * (width + 1) * disparity_range;
				cost_aggr_of_col += direction * (width + 1) * disparity_range;
				current_gray_col += direction * (width + 1);
			}

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void CostAggregateDagonal_2(const uint8* img_data, const sint32& width, const sint32& height,
	const sint32& min_disparity, const sint32& max_disparity, 
	const sint32& p1, const sint32& p2_init,
	const uint8* cost_init, uint8* cost_aggr, bool is_forward)
{
	if(!(width > 1 && height > 1 && max_disparity > min_disparity)){
		std::cout << "image/parameter error" << std::endl;
		return;
	}
	// 视差范围
	const sint32 disparity_range = max_disparity - min_disparity;

	// 正向(右上->左下) ：is_forward = true ; direction = 1
	// 反向(左下->右上) ：is_forward = false; direction = -1;
	const sint32 direction = is_forward ? 1 : -1;

	// 聚合
	
	// 存储当前的行列号，判断是否到达影像边界
	sint32 current_row = 0;
	sint32 current_col = 0;
//#pragma omp parallel for
	for (sint32 j = 0; j < width; j++) {
		uint8* cost_init_of_col;
		uint8* cost_aggr_of_col;
		uint8* current_gray_col;
		// 初始化赋值路径头
		if (is_forward) {
			// 指向第i+1行的第一个像素点灰度值
			current_gray_col = (uint8*)img_data + j;
			// 初始的代价取的都是第一个视差的
			cost_init_of_col = (uint8*)cost_init + j * disparity_range;
			cost_aggr_of_col = cost_aggr + j * disparity_range;
		}
		else {
			// 指向第i+2行第一个像素点的前面一个像素点 = 第i+1行最后一个
			current_gray_col = (uint8*)img_data + (height - 1) * width + j;
			cost_init_of_col = (uint8*)cost_init + (height - 1) * width * disparity_range + j * disparity_range;
			cost_aggr_of_col = cost_aggr + (height - 1) * width * disparity_range + j * disparity_range;
		}

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8> cost_last_pixel(disparity_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_of_col, cost_init_of_col, disparity_range * sizeof(uint8));
		memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));

		// 路径上当前灰度值和上一个灰度值
		uint8 gray = *current_gray_col;
		uint8 gray_last = *current_gray_col;

		// 对角线路径上的下一个像素，中间间隔width-1个像素
		// 这里要多一个边界处理
		// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == 0 && current_row < height - 1) {
			// 右上->左下，碰左边界
			cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
			cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
			current_gray_col = (uint8*)img_data + (current_row + direction) * width + (width - 1);
		}
		else if (!is_forward && current_col == width - 1 && current_row > 0) {
			// 左下->右上，碰右边界
			cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range;
			cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range;
			current_gray_col = (uint8*)img_data + (current_row + direction) * width;
		}
		else {
			cost_init_of_col += direction * (width - 1) * disparity_range;
			cost_aggr_of_col += direction * (width - 1) * disparity_range;
			current_gray_col += direction * (width - 1);
		}

		// 路径上上个像素的最小代价值
		uint8 mincost_last_pixel = UINT8_MAX;
		for (auto cost : cost_last_pixel) {
			mincost_last_pixel = std::min(mincost_last_pixel, cost);
		}

		// 自路径上第2个像素开始按顺序聚合
		for (sint32 i = 0; i < height - 1; i++) {
			gray = *current_gray_col;
			uint8 min_cost = UINT8_MAX;
			for (sint32 d = 0; d < disparity_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8  cost = cost_init_of_col[d];
				const uint16 l1 = cost_last_pixel[d + 1];
				const uint16 l2 = cost_last_pixel[d] + p1;
				const uint16 l3 = cost_last_pixel[d + 2] + p1;
				const uint16 l4 = mincost_last_pixel + p2_init / (abs(gray - gray_last) + 1);

				const uint8 cost_s = cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_pixel);

				cost_aggr_of_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_pixel = min_cost;
			memcpy(&cost_last_pixel[1], cost_aggr_of_col, disparity_range * sizeof(uint8));

			// 当前像素的行列号
			current_row += direction;
			current_col -= direction;

			// 下一个像素,这里要多一个边界处理
			// 这里要多一个边界处理
			// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
			if (is_forward && current_col == 0 && current_row < height - 1) {
				// 右上->左下，碰左边界
				cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
				cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range + (width - 1) * disparity_range;
				current_gray_col = (uint8*)img_data + (current_row + direction) * width + (width - 1);
			}
			else if (!is_forward && current_col == width - 1 && current_row > 0) {
				// 左下->右上，碰右边界
				cost_init_of_col = (uint8*)cost_init + (current_row + direction) * width * disparity_range;
				cost_aggr_of_col = cost_aggr + (current_row + direction) * width * disparity_range;
				current_gray_col = (uint8*)img_data + (current_row + direction) * width;
			}
			else {
				cost_init_of_col += direction * (width - 1) * disparity_range;
				cost_aggr_of_col += direction * (width - 1) * disparity_range;
				current_gray_col += direction * (width - 1);
			}

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

