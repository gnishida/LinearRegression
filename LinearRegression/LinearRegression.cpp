#include "LinearRegression.h"


LinearRegression::LinearRegression() {
}


void LinearRegression::train(const cv::Mat_<double>& X, const cv::Mat_<double>& Y) {
	W = X.inv(cv::DECOMP_SVD) * Y;
}

cv::Mat LinearRegression::predict(const cv::Mat_<double>& x) {
	return x * W;
}
