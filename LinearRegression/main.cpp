#include <stdio.h>
#include "MLUtils.h"
#include "LinearRegression.h"

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
	/*cv::Mat_<double> muX, maxX, muY, maxY;
	ml::normalizeDataset(X, X, muX, maxX);
	ml::normalizeDataset(Y, Y, muY, maxY);*/

	// テストデータの分割
	cv::Mat_<double> trainX, trainY, testX, testY;
	ml::splitDataset(X, 0.8f, trainX, testX);
	ml::splitDataset(Y, 0.8f, trainY, testY);

	ofstream ofs("results.txt");

	LinearRegression lr;
	//cv::Mat_<double> error;
	double rmse;
	double rmse_baseline;
	double residue;
	if (test_type == 0) {
		residue = lr.train(X, Y);

		cv::Mat Y_hat = lr.predict(X);
		/*for (int r = 0; r < Y_hat.rows; ++r) {
			cout << X.row(r) << endl;
			cout << "True, Pred" << endl;
			cout << Y.row(r) << endl;
			cout << Y_hat.row(r) << endl;
		}*/

		// 真値と予測値を保存
		ml::saveDataset("trueY.txt", Y);
		ml::saveDataset("predY.txt", Y_hat);

		rmse = ml::rmse(Y, Y_hat, true);

		cv::Mat Y_avg;
		cv::reduce(Y, Y_avg, 0, CV_REDUCE_AVG);
		rmse_baseline = ml::rmse(Y, cv::repeat(Y_avg, Y.rows, 1), true);
	} else {
		residue = lr.train(trainX, trainY);
		
		cv::Mat Y_hat = lr.predict(testX);
		/*for (int r = 0; r < Y_hat.rows; ++r) {
			cout << "True, Pred" << endl;
			cout << testY.row(r) << endl;
			cout << Y_hat.row(r) << endl;
		}*/

		// 真値と予測値を保存
		ml::saveDataset("trueY.txt", testY);
		ml::saveDataset("predY.txt", Y_hat);

		rmse = ml::rmse(testY, Y_hat, true);

		cv::Mat Y_avg;
		cv::reduce(trainY, Y_avg, 0, CV_REDUCE_AVG);
		rmse_baseline = ml::rmse(testY, cv::repeat(Y_avg, testY.rows, 1), true);
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
	cout << "Residue: " << residue << endl;
	cout << "-----------------------" << endl;
	cout << "RMSE: " << rmse << endl;
	cout << "Baselnie: " << rmse_baseline << endl;
	cout << endl;






	return 0;
}
