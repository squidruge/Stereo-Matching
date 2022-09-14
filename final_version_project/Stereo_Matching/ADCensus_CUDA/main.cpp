#include <iostream>
#include "ADCensusStereo.h"
#include <chrono>
#include "config.h"
#include <fstream> 
#include "performance_eval.h"


float32 cam_f[] = { 1733.74, 1734.04, 1740.10, 1739.18, 1758.23,
					1758.23, 1758.23, 1758.23, 1758.23, 1758.23,
					1733.68, 1734.16, 1742.11, 1742.17, 1729.48,
					1729.05, 1746.24, 1761.76, 1761.76, 1758.23,
					1758.23, 1758.23, 1769.02, 1765.39 };
float32 baseline[] = { 536.62, 529.50, 387.86, 573.76, 111.53,
					  124.86, 97.99,  78.28,  88.39,  56.46,
					  221.13, 228.38, 221.76, 237.66, 532.81,
					  537.75, 678.37, 119.36, 41.51,  87.85,
					  111.82, 76.83,  295.44, 294.57 };

const int doffs = 0;

void Disp2Depth2(const float32* disp_map, const sint32 width, const sint32 height, const std::string& path_save, const int num_of_picture);
using namespace std::chrono;


// opencv library
#include <opencv2/opencv.hpp>

/*��ʾ�Ӳ�ͼ*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*�����Ӳ�ͼ*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*�����Ӳ����*/
void SaveDisparityCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);


std::string ImgName[] = {	"artroom1","artroom2", "bandsaw1","bandsaw2",
						"chess1","chess2","chess3","curule1","curule2","curule3","ladder1","ladder2",
						"octogons1","octogons2","pendulum1","pendulum2","podium1","skates1","skates2",
						"skiboots1","skiboots2","skiboots3","traproom1","traproom2"};
std::string ImgName2003[] = {"cones","teddy"};
std::string ImgName2001[] = { "barn1","barn2","bull","map","poster","sawtooth","tsukuba","venus" };
std::string ImgName2022[] = { "my"};
const int Middlebury2021ImgNum = 24;


//���в�������Middbury2021 Middbury2003 Middbury2001
#if MIDDLEBURY2021ALL

int main(int argv, char** argc)
{
	//Ҫ���Ե�ͼƬ����Ĭ��Ϊ�������ݼ���С
	//const int RunNum = 1;
	const int RunNum = Middlebury2021ImgNum;
	float64 speed = 0.0;
	
	//���ݼ�·��
	//std::string dataset_path = "E:\\Program Files\\dataset\\Middlebury\\2003\\";
	//std::string disp_collect_path = "E:\\Program Files\\dataset\\Middlebury\\Disparity2003\\";

	std::string dataset_path = "E:\\Program Files\\dataset\\Middlebury\\2021\\";
	std::string disp_collect_path = "E:\\Program Files\\dataset\\Middlebury\\Disparity2021\\";

	//std::string dataset_path = "E:\\Program Files\\dataset\\Middlebury\\2001\\";
	//std::string disp_collect_path = "E:\\Program Files\\dataset\\Middlebury\\Disparity2001\\";
	
	//auto speed = 0.0f;


	//for (int ImgIdx = 0; ImgIdx < RunNum; ImgIdx++) {
	for (int ImgIdx = 0; ImgIdx < 1; ImgIdx++) {
		auto start_one = steady_clock::now();

		ImgIdx = 0;
		//����ͼ·��
		std::string path_left = "E:\\Program Files\\dataset\\Middlebury\\2022\\my\\im2.jpg";
		std::string path_right = "E:\\Program Files\\dataset\\Middlebury\\2022\\my\\im6.jpg";
		//std::string path_left = dataset_path+ ImgName[ImgIdx]+"\\im0.png";
		//std::string path_right = dataset_path + ImgName[ImgIdx] + "\\im1.png";

		//std::string path_left = dataset_path + ImgName2003[ImgIdx] + "\\im2.png";
		//std::string path_right = dataset_path + ImgName2003[ImgIdx] + "\\im6.png";

		//std::string path_left = dataset_path + ImgName2001[ImgIdx] + "\\im2.ppm";
		//std::string path_right = dataset_path + ImgName2001[ImgIdx] + "\\im6.ppm";
		// 
		//�Ӳ�ͼ�����ַ

		std::string disp_save_path = "E:\\Program Files\\dataset\\Middlebury\\2022\\my";
		//std::string disp_save_path = disp_collect_path + ImgName[ImgIdx];
		//std::string disp_save_path = disp_collect_path + ImgName2001[ImgIdx];
		//std::string disp_save_path = disp_collect_path + ImgName2003[ImgIdx];

		//std::string command;
		//command = "mkdir " + disp_save_path;
		//system(command.c_str());


		cv::Mat left_img = cv::imread(path_left);
		cv::Mat right_img = cv::imread(path_right);




		if (left_img.data == nullptr || right_img.data == nullptr) {
			std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
			return -1;
		}
		if (left_img.rows != right_img.rows || left_img.cols != right_img.cols) {
			std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
			return -1;
		}


		//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
		const sint32 width = static_cast<uint32>(left_img.cols);
		const sint32 height = static_cast<uint32>(right_img.rows);

		// ����Ӱ��Ĳ�ɫ����
		auto bytes_left = new uint8[width * height * 3];
		auto bytes_right = new uint8[width * height * 3];
#pragma omp parallel for 
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				bytes_left[i * 3 * width + 3 * j] = left_img.at<cv::Vec3b>(i, j)[0];
				bytes_left[i * 3 * width + 3 * j + 1] = left_img.at<cv::Vec3b>(i, j)[1];
				bytes_left[i * 3 * width + 3 * j + 2] = left_img.at<cv::Vec3b>(i, j)[2];
				bytes_right[i * 3 * width + 3 * j] = right_img.at<cv::Vec3b>(i, j)[0];
				bytes_right[i * 3 * width + 3 * j + 1] = right_img.at<cv::Vec3b>(i, j)[1];
				bytes_right[i * 3 * width + 3 * j + 2] = right_img.at<cv::Vec3b>(i, j)[2];
			}
		}
		//printf("Done!\n");

		// AD-Censusƥ��������
		ADCensusOption ad_option;
		// ��ѡ�ӲΧ
		ad_option.min_disparity = MIN_DISPARITY;
		ad_option.max_disparity = MAX_DISPARITY;
		// һ���Լ����ֵ
		ad_option.lrcheck_thres = LRCheckThres;

		// �Ƿ�ִ��һ���Լ��
		ad_option.do_lr_check = LRCheckOption;

		// �Ƿ�ִ���Ӳ����
		ad_option.do_filling = FillOption;

#if LOG
		printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, ad_option.min_disparity, ad_option.max_disparity);
#endif
		// ����AD-Censusƥ����ʵ��
		ADCensusStereo ad_census;

		//printf("AD-Census Initializing...\n");
		auto start = steady_clock::now();
		//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
		// ��ʼ��
		if (!ad_census.Initialize(width, height, ad_option)) {
			std::cout << "AD-Census��ʼ��ʧ�ܣ�" << std::endl;
			return -2;
		}
		auto end = steady_clock::now();
		auto tt = duration_cast<milliseconds>(end - start);
		//auto tt = duration_cast<milliseconds>(end - start);
#if LOG
		printf("AD-Census ��ʼ����ʱ:	%lf s\n\n", tt.count() / 1000.0);
#endif

		//printf("AD-Census Matching...\n");
		// disparity���鱣�������ص��Ӳ���
		auto disparity = new float32[uint32(width * height)]();

		start = steady_clock::now();
		//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
		// ƥ��
		if (!ad_census.Match(bytes_left, bytes_right, disparity)) {
			std::cout << "AD-Censusƥ��ʧ�ܣ�" << std::endl;
			return -2;
		}

		
#if SetBlackToZero
#pragma omp parallel for 
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (bytes_left[i * 3 * width + 3 * j] == 0 &&
					bytes_left[i * 3 * width + 3 * j + 1] == 0 &&
					bytes_left[i * 3 * width + 3 * j + 2] == 0) {
					disparity[i * width + j] = 0;
				}
				
		}
		}
#endif

		end = steady_clock::now();
		tt = duration_cast<milliseconds>(end - start);

#if LOG
		printf("\nAD-Census ƥ���ʱ :	%lf s\n", tt.count() / 1000.0);
#endif

		float64 time_matching = tt.count() / 1000.0;
		//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
		// ��ʾ�Ӳ�ͼ
		//ShowDisparityMap(disparity, width, height, "disp-left");


		/************** �ڴ˴����,������ĸ������ĳ�pfm�ļ���·���Ϳ�  ********************/
		// ����׼ȷ��
		//std::string gt_path = dataset_path + ImgName[ImgIdx] + "\\disp0.pfm";
		//PerformanceEval(disparity, width, height, gt_path, time_matching);

		//std::string gt_path = dataset_path + ImgName2001[ImgIdx] + "\\disp2.pgm";
		//PerformanceEval0103(disparity, width, height, gt_path, time_matching, 1);
		//std::string gt_path = dataset_path + ImgName2003[ImgIdx] + "\\disp2.png";
		//PerformanceEval2003(disparity, width, height, gt_path, time_matching);

		//����ͼ��ʱ
		auto end_one = steady_clock::now();
		auto time_one = duration_cast<seconds>(end_one - start_one);

		auto speed_one = width * height * MAX_DISPARITY * (0.000001) / float64(time_one.count());

		std::ofstream ofs;						//����������
		ofs.open(disp_save_path+"\\speed.txt", std::ios::out);		//��д�ķ�ʽ���ļ�
		ofs << speed_one << std::endl;//д��

		ofs.close();
		speed += speed_one;

		// �����Ӳ�ͼ
		SaveDisparityMap(disparity, width, height, disp_save_path);
		//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
		// 
		// 

		// �ͷ��ڴ�
		delete[] disparity;
		disparity = nullptr;
		delete[] bytes_left;
		bytes_left = nullptr;
		delete[] bytes_right;
		bytes_right = nullptr;
	}

	//����Ч����ƽ��
	speed /= RunNum;
	//printf("speed: %lf", speed);



	return 0;
}
#else



int main(int argv, char** argc)
{


#if MIDDLEBURY2003ONE

	std::string path_left = "E:\\Program Files\\Stereo Matching\\Stereo_Matching\\ADCensus_CUDA\\";
	//std::string img_path = "E:\\Program Files\\Stereo Matching\\Stereo_Matching\\ADCensus_CUDA";
	char Left_Origin_String[100];
	char Right_Origin_String[100];
	char Left_Disparity_String[100];
	char Right_Disparity_String[100];


	strcpy_s(Left_Origin_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Right_Origin_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Left_Disparity_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");
	strcpy_s(Right_Disparity_String, "E:\\Program Files\\Stereo Matching\\Middlebury\\");


	strcat_s(Left_Origin_String, "2003\\cones\\im2.png");
	strcat_s(Right_Origin_String, "2003\\cones\\im6.png");
	strcat_s(Left_Disparity_String, "2003\\cones\\disp2.png");
	strcat_s(Right_Disparity_String, "2003\\cones\\disp6.png");

	cv::Mat left_img = cv::imread(Left_Origin_String);
	cv::Mat right_img = cv::imread(Right_Origin_String);

#else if MIDDLEBURY2021ONE

	std::string path_left = "./im0.png";
	std::string path_right = "./im1.png";

	cv::Mat left_img = cv::imread(path_left);
	cv::Mat right_img = cv::imread(path_right);

#endif



	if (left_img.data == nullptr || right_img.data == nullptr) {
		std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
		return -1;
	}
	if (left_img.rows != right_img.rows || left_img.cols != right_img.cols) {
		std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
		return -1;
	}


	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	const sint32 width = static_cast<uint32>(left_img.cols);
	const sint32 height = static_cast<uint32>(right_img.rows);

	// ����Ӱ��Ĳ�ɫ����
	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = left_img.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = left_img.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = left_img.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = right_img.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = right_img.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = right_img.at<cv::Vec3b>(i, j)[2];
		}
	}
	//printf("Done!\n");

	// AD-Censusƥ��������
	ADCensusOption ad_option;
	// ��ѡ�ӲΧ
	ad_option.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
	ad_option.max_disparity = argv < 5 ? MaxDisparity : atoi(argc[4]);
	// һ���Լ����ֵ
	ad_option.lrcheck_thres = LRCheckThres;

	// �Ƿ�ִ��һ���Լ��
	ad_option.do_lr_check = LRCheckOption;

	// �Ƿ�ִ���Ӳ����
	// �Ӳ�ͼ���Ľ�������ɿ��������̣���������䣬�����У�������
	ad_option.do_filling = FillOption;
	
#if LOG
	printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, ad_option.min_disparity, ad_option.max_disparity);
#endif
	// ����AD-Censusƥ����ʵ��
	ADCensusStereo ad_census;

	//printf("AD-Census Initializing...\n");
	auto start = steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʼ��
	if (!ad_census.Initialize(width, height, ad_option)) {
		std::cout << "AD-Census��ʼ��ʧ�ܣ�" << std::endl;
		return -2;
	}
	auto end = steady_clock::now();
	auto tt = duration_cast<milliseconds>(end - start);
#if LOG
	printf("AD-Census ��ʼ����ʱ:	%lf s\n\n", tt.count() / 1000.0);
#endif

	//printf("AD-Census Matching...\n");
	// disparity���鱣�������ص��Ӳ���
	auto disparity = new float32[uint32(width * height)]();

	start = steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ƥ��
	if (!ad_census.Match(bytes_left, bytes_right, disparity)) {
		std::cout << "AD-Censusƥ��ʧ�ܣ�" << std::endl;
		return -2;
	}
	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);

#if 1
	printf("\nAD-Census ƥ���ʱ :	%lf s\n", tt.count() / 1000.0);
#endif
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʾ�Ӳ�ͼ
	ShowDisparityMap(disparity, width, height, "disp-left");
	std::string save_path = "..";
	// �����Ӳ�ͼ
	SaveDisparityMap(disparity, width, height, save_path);

	//�Ӳ�ͼת���ͼ
	Disp2Depth2(disparity, width, height, path_left, 4);

	cv::waitKey(0);

	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// �ͷ��ڴ�
	delete[] disparity;
	disparity = nullptr;
	delete[] bytes_left;
	bytes_left = nullptr;
	delete[] bytes_right;
	bytes_right = nullptr;

	system("pause");
	return 0;
}
#endif




void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name)
{
	// ��ʾ�Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(name, disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imshow(name + "-color", disp_color);

}

void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// �����Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}
	//printf("%s",path + "\\disparity.png");
	std::cout<< path + "\\disparity.png" <<std::endl;
	cv::imwrite(path + "\\disparity.png", disp_mat);
	//cv::imwrite("-d.png", disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	//cv::imwrite("-c.png", disp_color);
	cv::imwrite(path + "\\disparity_color.png", disp_color);
}

void SaveDisparityCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// �����Ӳ����(x,y,disp,r,g,b)
	FILE* fp_disp_cloud = nullptr;
	fopen_s(&fp_disp_cloud, (path + "-cloud.txt").c_str(), "w");
	if (fp_disp_cloud) {
		for (sint32 i = 0; i < height; i++) {
			for (sint32 j = 0; j < width; j++) {
				const float32 disp = abs(disp_map[i * width + j]);
				if (disp == Invalid_Float) {
					continue;
				}
				fprintf_s(fp_disp_cloud, "%f %f %f %d %d %d\n", float32(j), float32(i),
					disp, img_bytes[i * width * 3 + 3 * j + 2], img_bytes[i * width * 3 + 3 * j + 1], img_bytes[i * width * 3 + 3 * j]);
			}
		}
		fclose(fp_disp_cloud);
	}
}


void Disp2Depth2(const float32* disp_map, const sint32 width, const sint32 height, const std::string& path_save, const int num_of_picture)
{
	float32* depth_array = new float32[uint32(width * height)];
	memset(depth_array, 0, sizeof(float32) * width * height);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp = 0;
			}
			if (disp == 0)
				continue;
			depth_array[i * width + j] = (float32)(cam_f[num_of_picture] * baseline[num_of_picture] / disp + doffs);
		}
	}
	float32 max_depth = 0;
	float32 min_depth = 114514;
	// �������Сֵ
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 depth = abs(depth_array[i * width + j]);
			if (depth != Invalid_Float) {
				min_depth = std::min(min_depth, depth);
				max_depth = std::max(max_depth, depth);
				//std::cout << "max_disparity" << max_disp << std::endl;
			}
		}
	}
	std::cout << "max_depth" << max_depth << std::endl;
	std::cout << "min_depth" << min_depth << std::endl;

	const cv::Mat depth_mat = cv::Mat(height, width, CV_8UC1);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			float32 depth = abs(depth_array[i * width + j]);
			if (depth == Invalid_Float) {
				depth = 0;
			}
			else {
				depth_mat.data[i * width + j] = static_cast<uchar>((depth - min_depth) / (max_depth - min_depth) * 255);
			}

			
		}
	}

	cv::imwrite(path_save + "-depth.png", depth_mat);
	//ofstream���ڴ������ļ���ofstream��ʾ�ļ����������ζ��д�ļ�����
	std::ofstream outfile(path_save + "-depth.txt", std::ios::trunc);
	outfile << "max_depth: " << max_depth << std::endl;
	outfile << "min_depth: " << min_depth << std::endl;
	outfile.close();
}

