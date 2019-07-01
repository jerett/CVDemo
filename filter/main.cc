//
// Created by jerett on 2019-07-01.
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
using namespace cv;

struct GaussianFilterContext {
    Mat src_img;
    Mat dst_img;

    std::string win_name = "gaussian";
    int ksize = 1;
    const int kMaxSize = 30;
};

struct BilateralFilterContext {
    Mat src_img;
    Mat dst_img;

    std::string win_name = "bilateral";

    int d = 3;
    const int kMaxDSize = 30;

    int sigma_color = 10;
    const int kMaxSigmaColor = 200;

    int sigma_space = 10;
    const int kMaxSigmaSpace = 200;
};

void OnGaussianFilter(int, void *ctx) {
    auto gaussian_ctx = reinterpret_cast<GaussianFilterContext *>(ctx);

    int ksize = gaussian_ctx->ksize * 2 + 1;
    GaussianBlur(gaussian_ctx->src_img, gaussian_ctx->dst_img, Size(ksize, ksize), 0);
    imshow(gaussian_ctx->win_name, gaussian_ctx->dst_img);
}

void OnBilateralFilter(int, void *ctx) {
    auto bilateral_ctx = reinterpret_cast<BilateralFilterContext *>(ctx);
    // int ksize = gaussian_ctx->ksize * 2 + 1;
    // GaussianBlur(gaussian_ctx->src_img, gaussian_ctx->dst_img, Size(ksize, ksize), 0);
    // imshow(gaussian_ctx->win_name, gaussian_ctx->dst_img);
    bilateralFilter(bilateral_ctx->src_img,
                    bilateral_ctx->dst_img,
                    bilateral_ctx->d,
                    bilateral_ctx->sigma_color,
                    bilateral_ctx->sigma_space);
    imshow(bilateral_ctx->win_name, bilateral_ctx->dst_img);
}

void TestBilateral(BilateralFilterContext &ctx) {
    namedWindow(ctx.win_name, WINDOW_AUTOSIZE);

    std::string bar_name = "d: " + std::to_string(ctx.kMaxDSize);
    createTrackbar(bar_name, ctx.win_name, &ctx.d, ctx.kMaxDSize, OnBilateralFilter, &ctx);
    OnBilateralFilter(0, &ctx);

    bar_name = "sigma color: " + std::to_string(ctx.kMaxSigmaColor);
    createTrackbar(bar_name, ctx.win_name, &ctx.sigma_color, ctx.kMaxSigmaColor, OnBilateralFilter, &ctx);
    OnBilateralFilter(0, &ctx);

    bar_name = "sigma space: " + std::to_string(ctx.kMaxSigmaSpace);
    createTrackbar(bar_name, ctx.win_name, &ctx.sigma_space, ctx.kMaxSigmaSpace, OnBilateralFilter, &ctx);
    OnBilateralFilter(0, &ctx);
}

void TestGaussian(GaussianFilterContext &ctx) {
    namedWindow(ctx.win_name, WINDOW_AUTOSIZE);

    std::string bar_name = "kSize " + std::to_string(ctx.kMaxSize);
    createTrackbar(bar_name, ctx.win_name, &ctx.ksize, ctx.kMaxSize, OnGaussianFilter, &ctx);
    OnGaussianFilter(0, &ctx);
}

int main(int argc, char *argv[]) {
    auto test_img = imread(argv[1]);
    imshow("src", test_img);

    GaussianFilterContext gussian_ctx;
    gussian_ctx.src_img = test_img;
    TestGaussian(gussian_ctx);

    BilateralFilterContext bilateral_ctx;
    bilateral_ctx.src_img = test_img;
    TestBilateral(bilateral_ctx);
    waitKey(0);
    return 0;
}