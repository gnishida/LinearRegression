#include <stdio.h>
#include "MLUtils.h"
#include "LinearRegressionRegularization.h"

using namespace std;

int main(int argc,char *argv[]) {
	if (argc < 4) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename of X> <filename of Y> <test type>" << endl;
		cout << "       test type: 0 -- use the same data for training and test" << endl;
		cout << "       test type: 1 -- use 90% data for training and 10% for test" << endl;
		cout << endl;

		return -1;
	}

	int test_type = atoi(argv[3]);

	// テストデータを読み込む
	cv::Mat_<double> X;
	cv::Mat_<double> Y;
	if (!ml::loadDataset(argv[1], X)) {
		cout << "File " << argv[1] << " does not exist." << endl;
		return -1;
	}
	if (!ml::loadDataset(argv[2], Y)) {
		cout << "File " << argv[2] << " does not exist." << endl;
		return -1;
	}

	// テストデータのnormalize
	cv::Mat_<double> muX, maxX, muY, maxY;
	ml::normalizeDataset(X, X, muX, maxX);
	ml::normalizeDataset(Y, Y, muY, maxY);

	// バイアスの列を追加
	ml::addBias(X);

	// テストデータの分割
	cv::Mat_<double> trainX, trainY, testX, testY;
	ml::splitDataset(X, 0.9f, trainX, testX);
	ml::splitDataset(Y, 0.9f, trainY, testY);

	LinearRegressionRegularization lr;
	cv::Mat_<double> error;
	double residue;
	if (test_type == 0) {
		residue = lr.train(X, Y, 0.01, 0.1, 100);

		cv::Mat Y_hat = lr.predict(X);
		cv::reduce((Y - Y_hat).mul(Y - Y_hat), error, 0, CV_REDUCE_AVG);
	} else {
		residue = lr.train(trainX, trainY, 0.01, 0.1, 100);
		
		cv::Mat Y_hat = lr.predict(testX);
		cv::reduce((testY - Y_hat).mul(testY - Y_hat), error, 0, CV_REDUCE_AVG);
	}
	
	// テスト結果を出力
	cout << "Linear regression: " << argv[1] << endl;
	if (test_type == 0) {
		cout << "   (training data is used for test)" << endl;
	} else {
		cout << "   (90% for training data, 10% for test data)" << endl;
	}
	cout << "-----------------------" << endl;
	cout << "Condition number: " << lr.conditionNumber() << endl;
	cout << "-----------------------" << endl;
	cout << "Residue: " << residue << endl;
	cout << "-----------------------" << endl;
	cout << "Error: " << endl << error << endl;
	cout << endl;

	cv::reduce(error, error, 1, CV_REDUCE_SUM);

	cout << "-----------------------" << endl;
	cout << "RMSE: " << sqrt(error(0, 0)) << endl;
	cout << endl;

	return 0;
}
