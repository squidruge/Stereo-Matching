#include "performance_eval.h"
using namespace std;
using namespace cv;
 
/**
 * Loads a PFM image stored in little endian and returns the image as an OpenCV Mat.
 * @brief loadPFM
 * @param filePath
 * @return
 */
Mat LoadPFM(const std::string filePath)
{

    //Open binary file
    ifstream file(filePath.c_str(), ios::in | ios::binary);

    Mat imagePFM;

    //If file correctly openened
    if (file)
    {
        //Read the type of file plus the 0x0a UNIX return character at the end
        char type[3];
        file.read(type, 3 * sizeof(char));

        //Read the width and height
        unsigned int width(0), height(0);
        file >> width >> height;

        //Read the 0x0a UNIX return character at the end
        char endOfLine;
        file.read(&endOfLine, sizeof(char));

        int numberOfComponents(0);
        //The type gets the number of color channels
        if (type[1] == 'F')
        {
            imagePFM = Mat(height, width, CV_32FC3);
            numberOfComponents = 3;
        }
        else if (type[1] == 'f')
        {
            imagePFM = Mat(height, width, CV_32FC1);
            numberOfComponents = 1;
        }

        //TODO Read correctly depending on the endianness
        //Read the endianness plus the 0x0a UNIX return character at the end
        //Byte Order contains -1.0 or 1.0
        char byteOrder[4];
        file.read(byteOrder, 4 * sizeof(char));

        //Find the last line return 0x0a before the pixels of the image
        char findReturn = ' ';
        while (findReturn != 0x0a)
        {
            file.read(&findReturn, sizeof(char));
        }

        //Read each RGB colors as 3 floats and store it in the image.
        float* color = new float[numberOfComponents];
        for (unsigned int i = 0; i < height; ++i)
        {
            for (unsigned int j = 0; j < width; ++j)
            {
                file.read((char*)color, numberOfComponents * sizeof(float));

                //In the PFM format the image is upside down
                if (numberOfComponents == 3)
                {
                    //OpenCV stores the color as BGR
                    imagePFM.at<Vec3f>(height - 1 - i, j) = Vec3f(color[2], color[1], color[0]);
                }
                else if (numberOfComponents == 1)
                {
                    //OpenCV stores the color as BGR
                    imagePFM.at<float>(height - 1 - i, j) = color[0];
                }
            }
        }

        delete[] color;

        //Close file
        file.close();
    }
    else
    {
        cerr << "Could not open the file : " << filePath << endl;
    }

    return imagePFM;
}

/**
 * Saves the image as a PFM file.
 * @brief savePFM
 * @param image
 * @param filePath
 * @return
 */
bool savePFM(const cv::Mat image, const std::string filePath)
{
    //Open the file as binary!
    ofstream imageFile(filePath.c_str(), ios::out | ios::trunc | ios::binary);

    if (imageFile)
    {
        int width(image.cols), height(image.rows);
        int numberOfComponents(image.channels());

        //Write the type of the PFM file and ends by a line return
        char type[3];
        type[0] = 'P';
        type[2] = 0x0a;

        if (numberOfComponents == 3)
        {
            type[1] = 'F';
        }
        else if (numberOfComponents == 1)
        {
            type[1] = 'f';
        }

        imageFile << type[0] << type[1] << type[2];

        //Write the width and height and ends by a line return
        imageFile << width << " " << height << type[2];

        //Assumes little endian storage and ends with a line return 0x0a
        //Stores the type
        char byteOrder[10];
        byteOrder[0] = '-'; byteOrder[1] = '1'; byteOrder[2] = '.'; byteOrder[3] = '0';
        byteOrder[4] = '0'; byteOrder[5] = '0'; byteOrder[6] = '0'; byteOrder[7] = '0';
        byteOrder[8] = '0'; byteOrder[9] = 0x0a;

        for (int i = 0; i < 10; ++i)
        {
            imageFile << byteOrder[i];
        }

        //Store the floating points RGB color upside down, left to right
        float* buffer = new float[numberOfComponents];

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                if (numberOfComponents == 1)
                {
                    buffer[0] = image.at<float>(height - 1 - i, j);
                }
                else
                {
                    Vec3f color = image.at<Vec3f>(height - 1 - i, j);

                    //OpenCV stores as BGR
                    buffer[0] = color.val[2];
                    buffer[1] = color.val[1];
                    buffer[2] = color.val[0];
                }

                //Write the values
                imageFile.write((char*)buffer, numberOfComponents * sizeof(float));

            }
        }

        delete[] buffer;

        imageFile.close();
    }
    else
    {
        cerr << "Could not open the file : " << filePath << endl;
        return false;
    }

    return true;
}

// disp_estimated使用计算出的原始视差数组，不需要归一化
void PerformanceEval(const float32* disp_estimated, const sint32& width, const sint32& height, const std::string& GT_path, const float32& time) {
    
    // 载入数据
    Mat disp_GT = LoadPFM(GT_path);
    imwrite(GT_path + "-saved.png", disp_GT);

    float32 RMS = 0.0f, PEP = 0.0f;
    float32 Mde = 0.0f, d_max = 0.0f, disp = 0.0f;
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
            d_max = max(d_max, disp_estimated[i * width + j]);
            if (disp_GT.at<float32>(i, j) == Invalid_Float)
                disp = 0;
            else
                disp = disp_GT.at<float32>(i, j);
			const float32 delta = disp_estimated[i * width + j] - disp;
			RMS += delta * delta;
			PEP += (abs(delta) > delta_d);
            //cout <<"("<< i << "," << j << ")" <<"  "<< disp_estimated[i * width + j] << "   " << disp << endl;
		}
	}

    // 准确度评估
	RMS = sqrt(RMS / (height * width));
	PEP /= (height * width);

    // 效率评估
    Mde = (width * height * d_max * (1e-6)) / time;

    printf("\nEvalution:\n  RMS = %.3f\n  PEP = %.3f (bad %.1f)\n  Mde = %.3f\n", RMS, PEP, delta_d, Mde);

}

//对2001 2003数据集使用
void PerformanceEval0103(const float32* disp_estimated, const sint32& width, const sint32& height, const std::string& GT_path, const float64& time, const sint32 type) {
    // 载入数据
    Mat disp_GT = imread(GT_path, 0);
    //imwrite(GT_path + "-saved.png", disp_GT);

    float32 RMS = 0.0f, PEP = 0.0f;
    float32 Mde = 0.0f, d_max = 0.0f, disp = 0.0f;
    float32 delta = 0.0f;
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            d_max = max(d_max, disp_estimated[i * width + j]);
            disp = disp_GT.at<uchar>(i, j);
            if (disp == 0) {
                continue;
            }
            if (disp_estimated[i * width + j] != Invalid_Float) {
                delta = disp_estimated[i * width + j] - disp / para[type];
            }else{
                delta = 0 - disp / para[type];
            }
            
            RMS += delta * delta;
            PEP += (abs(delta) > delta_d);
           //cout << disp_estimated[i * width + j] /disp << endl;
        }
    }

    // 准确度评估
    RMS = sqrt(RMS / (height * width));
    PEP /= (height * width);

    // 效率评估
    Mde = (width * height * d_max  / time*(1e-6)) ;

    printf("\nEvalution:\n  RMS = %.3f\n  PEP = %.3f (bad %.1f)\n  Mde = %.3f\n", RMS, PEP, delta_d, Mde);
}