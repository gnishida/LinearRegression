#include <stdio.h>
#include "MLUtils.h"
#include "LinearRegression.h"

using namespace std;

int main(int argc,char *argv[]) {
	if (argc < 3) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename> <test type>" << endl;
		cout << "       test type: 0 -- use the same data for training and test" << endl;
		cout << "       test type: 1 -- use 80% data for training and 20% for test" << endl;
		cout << endl;

		return -1;
	}

	int test_type = atoi(argv[2]);

	// テストデータを読み込む
	cv::Mat_<double> X;
	cv::Mat_<double> Y;
	if (!ml::loadDataset(argv[1], X, Y)) {
		cout << "File does not exist." << endl;
		return -1;
	}

	// テストデータのnormalize
	cv::Mat_<double> muX, maxX, muY, maxY;
	ml::normalizeDataset(X, X, muX, maxX);
	ml::normalizeDataset(Y, Y, muY, maxY);

	// テストデータの分割
	cv::Mat_<double> trainX, trainY, testX, testY;
	cv::Mat_<double> validX, validY;
	//ml::splitDataset(X, 0.9f, trainX, testX);
	//ml::splitDataset(Y, 0.9f, trainY, testY);
	ml::splitDataset(X, 0.8f, 0.1f, trainX, validX, testX);
	ml::splitDataset(Y, 0.8f, 0.1f, trainY, validY, testY);

	LinearRegression lr;
	cv::Mat_<double> error;
	if (test_type == 0) {
		lr.train(X, Y);

		cv::Mat Y_hat = lr.predict(X);
		cv::reduce((Y - Y_hat).mul(Y - Y_hat), error, 0, CV_REDUCE_AVG);
		cv::sqrt(error, error);
	} else {
		lr.train(trainX, trainY);
		
		cv::Mat Y_hat = lr.predict(testX);
		cv::reduce((testY - Y_hat).mul(testY - Y_hat), error, 0, CV_REDUCE_AVG);
		cv::sqrt(error, error);
	}
	
	// テスト結果を出力
	cout << "Linear regression: " << argv[1] << endl;
	if (test_type == 0) {
		cout << "   (training data is used for test)" << endl;
	} else {
		cout << "   (80% for training data, 20% for test data)" << endl;
	}
	cout << "-----------------------" << endl;
	cout << "Condition number: " << lr.conditionNumber() << endl;
	cout << "-----------------------" << endl;
	cout << "Error: " << endl << error << endl;
	cout << endl;

	cv::reduce(error, error, 1, CV_REDUCE_AVG);

	cout << "-----------------------" << endl;
	cout << "Avg error: " << error(0, 0) << endl;
	cout << endl;

	return 0;
}
