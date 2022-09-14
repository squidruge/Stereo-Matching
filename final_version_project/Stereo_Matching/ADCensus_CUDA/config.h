#pragma once

#define MaxDisparity 130
#define MinDispMismatch 0
#define LOG true

#define MIDDLEBURY2003ONE false
#define MIDDLEBURY2021ONE false
#define MIDDLEBURY2021ALL true


#define LAMBDA_AD 10
#define LAMBDA_CENSUS 30

//refiner
//Î¨Ò»ÐÔÔ¼Êø
#define is_check_unique false
#define uniqueness_ratio 0.9935//0.9935

#define OcclusionsRefine true
#define MismatchesRefine true


#define MaxSearchLengthTimes 1

#define CrossL1  34 //34
#define CrossL2 17  //17
#define CrossTau1 20 //20
#define CrossTau2 6  //6


//chess 2 no check
//#define MinDispMismatch 50
//#define MaxSearchLengthTimes 0.5 



#define MAX_DISPARITY MaxDisparity
#define MIN_DISPARITY 0

#define CENSUS_WIDTH		9
#define CENSUS_HEIGHT		7
#define TOP				(CENSUS_HEIGHT-1)/2
#define LEFT			(CENSUS_WIDTH-1)/2

#define IterativeNums 5

#define WARP_SIZE		32


#define FERMI false

#define GPU_THREADS_PER_BLOCK_FERMI 256
#define GPU_THREADS_PER_BLOCK_MAXWELL 64

/* Defines related to GPU Architecture */
#if FERMI
#define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_FERMI
#else
#define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_MAXWELL
#endif

#define BLOCK_SIZE					256
#define COSTAGG_BLOCKSIZE			GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ		GPU_THREADS_PER_BLOCK



//fuction

#define MIN(a,b) (a<b)?(a):(b)
#define MAX(a,b) (a>b)?(a):(b)




#define LRCheckOption true
#define LRCheckThres  1
#define RegionVotingOption  true
#define InterpolatingOption true
#define DiscontinuityAdjustmentOption true
#define FillOption true






#define ScanlineOption true


#define SetBlackToZero false