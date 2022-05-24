#pragma once
#include <cstdint>

typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

class SemiGlobalMatching
{
public:
	SemiGlobalMatching();
	~SemiGlobalMatching();

	enum CensusSize {
		Census5x5 = 0,
		Census9x7
	};


	/** \brief SGM参数结构体 */
	struct SGMOption {
		uint8	num_paths;		// 聚合路径数
		sint32  min_disparity;	// 最小视差
		sint32	max_disparity;	// 最大视差
		CensusSize census_size;
		// P1,P2 
		// P2 = P2_int / (Ip-Iq)
		sint32  p1;				// 惩罚项参数P1
		sint32  p2_int;			// 惩罚项参数P2

		SGMOption() : num_paths(8), min_disparity(0), max_disparity(64), p1(10), p2_int(150) {
		}

	};
public:
	/**
	 * \brief 类的初始化，完成一些内存的预分配、参数的预设置等
	 * \param width		输入，核线像对影像宽
	 * \param height	输入，核线像对影像高
	 * \param option	输入，SemiGlobalMatching参数
	 */
	bool Initialize(const uint32& width, const uint32& height, const SGMOption& option);

	/**
	 * \brief 执行匹配
	 * \param img_left		输入，左影像数据指针
	 * \param img_right		输入，右影像数据指针
	 * \param disp_left		输出，左影像深度图指针，预先分配和影像等尺寸的内存空间
	 */
	bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

	/**
	 * \brief 重设
	 * \param width		输入，核线像对影像宽
	 * \param height	输入，核线像对影像高
	 * \param option	输入，SemiGlobalMatching参数
	 */
	bool Reset(const uint32& width, const uint32& height, const SGMOption& option);

	void Release();
	bool Initialize(const sint32& width, const sint32& height, const SGMOption& option);
private:

	/** \brief Census变换 */
	void CensusTransform() const;

	/** \brief 代价计算	 */
	void ComputeCost() const;

	/** \brief 代价聚合	 */
	void CostAggregation() const;

	/** \brief 视差计算	 */
	void ComputeDisparity() const;

	/** \brief 一致性检查 */
	void LRCheck() const;

private:
	/** \brief SGM参数	 */
	SGMOption option_;

	/** \brief 影像宽	 */
	sint32 width_;

	/** \brief 影像高	 */
	sint32 height_;

	/** \brief 左影像数据	 */
	uint8* img_left_;

	/** \brief 右影像数据	 */
	uint8* img_right_;

	/** \brief 左影像census值	*/
	uint32* census_left_;

	/** \brief 右影像census值	*/
	uint32* census_right_;

	/** \brief 初始匹配代价	*/
	uint8* cost_init_;

	/** \brief 聚合匹配代价	*/
	uint16* cost_aggr_;

	/** \brief 左影像视差图	*/
	float32* disp_left_;
	float32* disp_right_;

	/** \brief 是否初始化标志	*/
	bool is_initialized_;

	uint8* cost_aggr_1_;
	uint8* cost_aggr_2_;
	uint8* cost_aggr_3_;
	uint8* cost_aggr_4_;
	uint8* cost_aggr_5_;
	uint8* cost_aggr_6_;
	uint8* cost_aggr_7_;
	uint8* cost_aggr_8_;


	std::vector<std::pair<int, int>> occlusions_;

	std::vector<std::pair<int, int>> mismatches_;
};
