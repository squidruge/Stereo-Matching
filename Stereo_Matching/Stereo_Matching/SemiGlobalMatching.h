#pragma once
#include <cstdint>

typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

class SemiGlobalMatching
{
public:
	SemiGlobalMatching();
	~SemiGlobalMatching();

	/** \brief SGM�����ṹ�� */
	struct SGMOption {
		uint8	num_paths;		// �ۺ�·����
		sint32  min_disparity;	// ��С�Ӳ�
		sint32	max_disparity;	// ����Ӳ�

		// P1,P2 
		// P2 = P2_int / (Ip-Iq)
		sint32  p1;				// �ͷ������P1
		sint32  p2_int;			// �ͷ������P2

		SGMOption() : num_paths(8), min_disparity(0), max_disparity(64), p1(10), p2_int(150) {
		}

	};
public:
	/**
	 * \brief ��ĳ�ʼ�������һЩ�ڴ��Ԥ���䡢������Ԥ���õ�
	 * \param width		���룬�������Ӱ���
	 * \param height	���룬�������Ӱ���
	 * \param option	���룬SemiGlobalMatching����
	 */
	bool Initialize(const uint32& width, const uint32& height, const SGMOption& option);

	/**
	 * \brief ִ��ƥ��
	 * \param img_left		���룬��Ӱ������ָ��
	 * \param img_right		���룬��Ӱ������ָ��
	 * \param disp_left		�������Ӱ�����ͼָ�룬Ԥ�ȷ����Ӱ��ȳߴ���ڴ�ռ�
	 */
	bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

	/**
	 * \brief ����
	 * \param width		���룬�������Ӱ���
	 * \param height	���룬�������Ӱ���
	 * \param option	���룬SemiGlobalMatching����
	 */
	bool Reset(const uint32& width, const uint32& height, const SGMOption& option);

private:

	/** \brief Census�任 */
	void CensusTransform() const;

	/** \brief ���ۼ���	 */
	void ComputeCost() const;

	/** \brief ���۾ۺ�	 */
	void CostAggregation() const;

	/** \brief �Ӳ����	 */
	void ComputeDisparity() const;

	/** \brief һ���Լ�� */
	void LRCheck() const;

private:
	/** \brief SGM����	 */
	SGMOption option_;

	/** \brief Ӱ���	 */
	sint32 width_;

	/** \brief Ӱ���	 */
	sint32 height_;

	/** \brief ��Ӱ������	 */
	uint8* img_left_;

	/** \brief ��Ӱ������	 */
	uint8* img_right_;

	/** \brief ��Ӱ��censusֵ	*/
	uint32* census_left_;

	/** \brief ��Ӱ��censusֵ	*/
	uint32* census_right_;

	/** \brief ��ʼƥ�����	*/
	uint8* cost_init_;

	/** \brief �ۺ�ƥ�����	*/
	uint16* cost_aggr_;

	/** \brief ��Ӱ���Ӳ�ͼ	*/
	float32* disp_left_;

	/** \brief �Ƿ��ʼ����־	*/
	bool is_initialized_;
};
