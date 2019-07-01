//
// Created by jerett on 2019-05-08.
//

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

void sobel(const cv::Mat &img) {
    img.convertTo(img, CV_32F, 1 / 255.0);
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, true);
    imshow("img", img);
    imshow("gx", gx);
    imshow("gy", gy);
    imshow("mag", mag);

    waitKey(0);
}

Mat get_hogdescriptor_visual_image(Mat &origImg,
                                   vector<float> &descriptorValues,//hog特征向量
                                   Size winSize,//图片窗口大小
                                   Size cellSize,
                                   int scaleFactor,//缩放背景图像的比例
                                   double viz_factor)//缩放hog特征的线长比例
{
    Mat visual_image;//最后可视化的图像大小
    resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14 / (float) gradientBinSize; //pi=3.14对应180°

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = winSize.width / cellSize.width;//x方向上的cell个数
    int cells_in_y_dir = winSize.height / cellSize.height;//y方向上的cell个数
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;//cell的总个数
    //注意此处三维数组的定义格式
    //int ***b;
    //int a[2][3][4];
    //int (*b)[3][4] = a;
    //gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]
    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;//把每个cell的9个bin对应的梯度强度都初始化为0
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    //相当于blockstride = (8,8)
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
        for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
            // 4 cells per block ...
            for (int cellNr = 0; cellNr < 4; cellNr++) {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr == 1) celly++;
                if (cellNr == 2) cellx++;
                if (cellNr == 3) {
                    cellx++;
                    celly++;
                }

                for (int bin = 0; bin < gradientBinSize; bin++) {
                    float gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;//因为C是按行存储

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;//由于block之间有重叠，所以要记录哪些cell被多次计算了

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < gradientBinSize; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }

    cout << "winSize = " << winSize << endl;
    cout << "cellSize = " << cellSize << endl;
    cout << "blockSize = " << cellSize * 2 << endl;
    cout << "blockNum = " << blocks_in_x_dir << "×" << blocks_in_y_dir << endl;
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // draw cells
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;

            int mx = drawX + cellSize.width / 2;
            int my = drawY + cellSize.height / 2;

            rectangle(visual_image,
                      Point(drawX * scaleFactor, drawY * scaleFactor),
                      Point((drawX + cellSize.width) * scaleFactor,
                            (drawY + cellSize.height) * scaleFactor),
                      CV_RGB(0, 0, 0),//cell框线的颜色
                      1);

            // draw in each cell all 9 gradient strengths
            for (int bin = 0; bin < gradientBinSize; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;//取每个bin里的中间值，如10°,30°,...,170°.

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cellSize.width / 2;
                float scale = viz_factor; // just a visual_imagealization scale,
                // to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     Point(x1 * scaleFactor, y1 * scaleFactor),
                     Point(x2 * scaleFactor, y2 * scaleFactor),
                     CV_RGB(255, 255, 255),//HOG可视化的cell的颜色
                     1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visual_image;//返回最终的HOG可视化图像
}

void hog(const Mat &img) {
    HOGDescriptor hog_desc;
    int align_width = ((img.cols - 1) / hog_desc.blockSize.width + 1) * hog_desc.blockSize.width;
    int align_height = ((img.rows - 1) / hog_desc.blockSize.height + 1) * hog_desc.blockSize.height;
    Size align_size(align_width, align_height);
    CV_LOG_INFO(NULL, "resize img size from:" << Size(img.cols, img.rows) << " to size:" << align_size);
    hog_desc.winSize = align_size;
    CV_LOG_INFO(NULL, "hog desc bins:" << hog_desc.nbins
                                       << " block size:" << hog_desc.blockSize
                                       << " block stride:" << hog_desc.blockStride
                                       << " cell size:" << hog_desc.cellSize);

    Mat hog_img;
    resize(img, hog_img, align_size);

    vector<float> descriptors;
    vector<Point> locations;
    hog_desc.compute(hog_img, descriptors, Size(0, 0), Size(0, 0), locations);

    // visualize
    Mat background = Mat::zeros(align_size, CV_8UC1);
    Mat visualize_hog_feature = get_hogdescriptor_visual_image(background, descriptors, align_size, hog_desc.cellSize, 1, 1);

    imshow("img", hog_img);
    imshow("hog feauture", visualize_hog_feature);
    waitKey();
    CV_LOG_INFO(NULL, "desc size:" << descriptors.size());
}

int main(int argc, char *argv[]) {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_DEBUG);
    Mat img = imread("person.png");
    CV_LOG_INFO(NULL, "img size:" << img.size);
    // sobel(img);
    hog(img);
    return 0;
}
