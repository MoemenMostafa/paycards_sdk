//
//  NumberRecognizer.cpp
//  CardRecognizer
//
//  Created by Vladimir Tchernitski on 12/01/16.
//  Copyright © 2016 Vladimir Tchernitski. All rights reserved.
//
#include <sstream> // string stream
#include <iterator> // ostream_iterator
#include <iostream> // cout
#include <android/log.h>
#include "NumberRecognizer.h"
#include "IServiceContainer.h"
#include "IRecognitionCore.h"
#include "INeuralNetworkObjectFactory.h"
#include "IRecognitionCoreDelegate.h"
#include "Utils.h"

static const cv::Rect dateWindowRect(257,282,210,65);
// Visa, Master, Dinner
static const cv::Rect numberWindowRect(54,221,552,60);
// American
static const cv::Rect numberWindowAmericanRect(79,221,552,60);
// Card number Visa,Master
static const cv::Rect areaX0(0,0,136,37);
static const cv::Rect areaX1(138,0,136,37);
static const cv::Rect areaX2(277,0,136,37);
static const cv::Rect areaX3(416,0,136,37);
// Card number Diner
static const cv::Rect areaY0(0,0,136,37);
static const cv::Rect areaY1(138,0,204,37);
static const cv::Rect areaY2(345,0,136,37);
// Card number American
static const cv::Rect areaZ0(0,0,136,37);
static const cv::Rect areaZ1(138,0,204,37);
static const cv::Rect areaZ2(342,0,136,37);

static const vector<cv::Rect> areasX = {areaX0, areaX1, areaX2, areaX3};
static const vector<cv::Rect> areasY = {areaY0, areaY1, areaY2};
static const vector<cv::Rect> areasZ = {areaZ0, areaZ1, areaZ2};

static const cv::Size digitSize(25,37); // 960/660 = 1.45454545
static int panDigitPaddingX = 2;
static int panDigitPaddingY = 2;
static const int spaceBWDidits = 3;

int scanCount;

CNumberRecognizer::CNumberRecognizer(const shared_ptr<IServiceContainer>& container) : _container(container)
{
    if(auto container = _container.lock()) {
        _factory = container->resolve<INeuralNetworkObjectFactory>();
    }
}

CNumberRecognizer::~CNumberRecognizer()
{
}

void CNumberRecognizer::SetRecognitionMode(PayCardsRecognizerMode flag)
{
    _mode = flag;
}

void CNumberRecognizer::SetDelegate(const shared_ptr<IRecognitionCoreDelegate>& delegate)
{
    _delegate = delegate;
}

void CNumberRecognizer::SetPathNumberRecognitionModel(const string& path)
{
    _pathNumberRecognitionModel = path;
}

void CNumberRecognizer::SetPathNumberRecognitionStruct(const string& path)
{
    _pathNumberRecognitionStruct = path;
}

void CNumberRecognizer::SetPathNumberLocalizationXModel(const string& path)
{
    _pathNumberLocalizationXModel = path;
}

void CNumberRecognizer::SetPathNumberLocalizationXStruct(const string& path)
{
    _pathNumberLocalizationXStruct = path;
}

void CNumberRecognizer::SetPathNumberLocalizationYModel(const string& path)
{
    _pathNumberLocalizationYModel = path;
}

void CNumberRecognizer::SetPathNumberLocalizationYStruct(const string& path)
{
    _pathNumberLocalizationYStruct = path;
}

bool CNumberRecognizer::Deploy()
{
    if(auto factory = _factory.lock()) {
        _recognitionNeuralNetwork = factory->CreateNeuralNetwork("", _pathNumberRecognitionStruct, _pathNumberRecognitionModel);
        
        _localizationNeuralNetworkY = factory->CreateNeuralNetwork("", _pathNumberLocalizationYStruct, _pathNumberLocalizationYModel);
        
        _localizationNeuralNetworkX = factory->CreateNeuralNetwork("", _pathNumberLocalizationXStruct, _pathNumberLocalizationXModel);
        
        return _localizationNeuralNetworkY->IsDeployed() &&
                _localizationNeuralNetworkX->IsDeployed() && _recognitionNeuralNetwork->IsDeployed();
    }
    
    return false;
}

///////////////////////////////////////////////////////////////////////

Mat CNumberRecognizer::HistY(const Mat& blockMat)
{
    Mat sobelY;
    Sobel(blockMat, sobelY, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    
    Mat absSobelY;
    convertScaleAbs( sobelY, absSobelY );
    
    cv::Mat histY(cv::Mat::zeros(1,numberWindowRect.height,CV_32FC1));
    cv::reduce(absSobelY, histY, 1, CV_REDUCE_SUM, CV_32FC1);
    
    Mat result;
    cv::normalize(histY, result, 0, 255, NORM_MINMAX, CV_8UC1);
    
    return result;
}

void CNumberRecognizer::Predict(const vector<Mat>& matrixes, shared_ptr<INeuralNetworkResultList>& neuralNetworkResultList, const shared_ptr<INeuralNetwork>& neuralNetwork)
{
    if(auto factory = _factory.lock()) {
        shared_ptr<INeuralNetworkDatumList> neuralNetworkDatumList = factory->CreateNeuralNetworkDatumList();
        
        for(Mat matrix : matrixes) {
            shared_ptr<INeuralNetworkDatum> neuralNetworkDatum = factory->CreateNeuralNetworkDatum(matrix);
            neuralNetworkDatumList->PushBack(neuralNetworkDatum);
        }
        
        neuralNetwork->Predict(neuralNetworkDatumList, neuralNetworkResultList);
    }
}

vector<Mat> CNumberRecognizer::SplitBlock(const Mat& mat, int xPos, int yPos, int offset,
                                        int digitSpace, int xPadding, int yPadding, vector<cv::Rect>& digitsRects)
{
    vector<Mat> digits;
    
    int x = offset + xPos;
    
    for(int i=0; i<4; i++) {
        
        cv::Rect rect = cv::Rect(x-xPadding, yPos-yPadding, digitSize.width + xPadding*2, digitSize.height + yPadding*2);
        
        if (CUtils::ValidateROI(mat, rect)) {
            digitsRects.push_back(rect);
            Mat digit = mat(rect);
            digits.push_back(digit);
            x += digitSize.width + digitSpace;
        }
    }
    
    return digits;
}

vector<Mat> CNumberRecognizer::SplitBlock6(const Mat& mat, int xPos, int yPos, int offset,
                                          int digitSpace, int xPadding, int yPadding, vector<cv::Rect>& digitsRects)
{
    vector<Mat> digits;

    int x = offset + xPos;

    for(int i=0; i<6; i++) {

        cv::Rect rect = cv::Rect(x-xPadding, yPos-yPadding, digitSize.width + xPadding*2, digitSize.height + yPadding*2);

        if (CUtils::ValidateROI(mat, rect)) {
            digitsRects.push_back(rect);
            Mat digit = mat(rect);
            digits.push_back(digit);
            x += digitSize.width + digitSpace;
        }
    }

    return digits;
}

vector<Mat> CNumberRecognizer::SplitBlock5(const Mat& mat, int xPos, int yPos, int offset,
                                           int digitSpace, int xPadding, int yPadding, vector<cv::Rect>& digitsRects)
{
    vector<Mat> digits;

    int x = offset + xPos;

    for(int i=0; i<5; i++) {

        cv::Rect rect = cv::Rect(x-xPadding, yPos-yPadding, digitSize.width + xPadding*2, digitSize.height + yPadding*2);

        if (CUtils::ValidateROI(mat, rect)) {
            digitsRects.push_back(rect);
            Mat digit = mat(rect);
            digits.push_back(digit);
            x += digitSize.width + digitSpace;
        }
    }

    return digits;
}

shared_ptr<INeuralNetworkResultList> CNumberRecognizer::Process(cv::Mat& matrix, cv::Rect& boundingRect, int scanTime)
{
    scanCount = scanTime;
    Mat numberWindow;
    cv::Rect extendedRect;
    const int padding = 10;

    if(scanTime == 1){
        numberWindow = matrix(numberWindowAmericanRect);
        extendedRect = cv::Rect(numberWindowAmericanRect.x - padding, numberWindowAmericanRect.y - padding,
                                numberWindowAmericanRect.width + padding*2, numberWindowAmericanRect.height + padding*2);
    } else {
        numberWindow = matrix(numberWindowRect);
        extendedRect = cv::Rect(numberWindowRect.x - padding, numberWindowRect.y - padding,
                                numberWindowRect.width + padding*2, numberWindowRect.height + padding*2);
    }

    Mat extendedNumberWindow = matrix(extendedRect);
    
    vector<cv::Point> points = {cv::Point(0,0),cv::Point(0,0),cv::Point(0,0),cv::Point(0,0)};
    
    if (PreLocalize(numberWindow, matrix, points)) {
        
        for (cv::Point& point : points) {
            point += cv::Point(padding,padding);
        }
        shared_ptr<INeuralNetworkResultList> result = ProcessMatrixFinal(extendedNumberWindow, points, _recognitionNeuralNetwork, cv::Point(panDigitPaddingX,panDigitPaddingY), boundingRect);
        
        boundingRect.x += extendedRect.x;
        boundingRect.y += extendedRect.y;
        
        return result;
    }
    
    return nullptr;
}


bool CNumberRecognizer::PreLocalize(Mat& numberWindow, Mat& matrix, vector<cv::Point>& points)
{
    if(auto factory = _factory.lock()) {
        Mat histY = HistY(numberWindow);
        
        shared_ptr<INeuralNetworkResultList> neuralNetworkResultListY = factory->CreateNeuralNetworkResultList();
        
        Predict({histY}, neuralNetworkResultListY, _localizationNeuralNetworkY);
        
        shared_ptr<INeuralNetworkResult> resultY = neuralNetworkResultListY->GetAtIndex(0);
        vector<pair<int, float>> data = resultY->GetRawResult();
        
        for (cv::Point& point : points) {
            point.y = cvRound(data.at(0).second*23.0);
        }
        Rect rect;
        if(scanCount == 1){
            rect = cv::Rect(numberWindowAmericanRect.x, numberWindowAmericanRect.y + points[0].y, numberWindowAmericanRect.width, digitSize.height);
        } else {
            rect = cv::Rect(numberWindowRect.x, numberWindowRect.y + points[0].y, numberWindowRect.width, digitSize.height);
        }
        if (!CUtils::ValidateROI(matrix, rect)) return false;
        
        Mat fullNumberMat = matrix(rect);
        
        shared_ptr<INeuralNetworkResultList> neuralNetworkResultListX = _factory.lock()->CreateNeuralNetworkResultList();
        
        vector<Mat> blocks;

        if(scanCount == 1){
            for(cv::Rect rect : areasZ) {
                Rect _rect = cv::Rect(rect.x,0, rect.width,fullNumberMat.rows);

                if (!CUtils::ValidateROI(fullNumberMat, _rect)) return false;
                Mat block = fullNumberMat(_rect);

                blocks.push_back( block );
            }
        } else if(scanCount == 2){
            for(cv::Rect rect : areasX) {
                Rect _rect = cv::Rect(rect.x,0, rect.width,fullNumberMat.rows);

                if (!CUtils::ValidateROI(fullNumberMat, _rect)) return false;
                Mat block = fullNumberMat(_rect);

                blocks.push_back( block );
            }
        } else {
            for(cv::Rect rect : areasY) {
                Rect _rect = cv::Rect(rect.x,0, rect.width,fullNumberMat.rows);

                if (!CUtils::ValidateROI(fullNumberMat, _rect)) return false;
                Mat block = fullNumberMat(_rect);

                blocks.push_back( block );
            }
        }

        Predict(blocks, neuralNetworkResultListX, _localizationNeuralNetworkX);
        
        int k=0;
        for (auto it = neuralNetworkResultListX->Begin(); it != neuralNetworkResultListX->End(); ++it, ++k) {
            
            shared_ptr<INeuralNetworkResult> resultItem = *it;
            vector<pair<int, float>> data = resultItem->GetRawResult();
            
            points[k].x = cvRound(data.at(0).second*24) + 2; // magic +2
        }
        
        return true;
    }
    
    return false;
}

shared_ptr<INeuralNetworkResultList> CNumberRecognizer::ProcessMatrixFinal(Mat& numberWindow,
                                                                           const vector<cv::Point>& points,
                                                                           const shared_ptr<INeuralNetwork>& neuralNetwork,
                                                                           cv::Point paddingPoint, cv::Rect& boundingRect)
{
    if(auto factory = _factory.lock()) {
        vector<Mat> digits;

        int count = 0;
        
        vector<cv::Rect> digitRects;

        if(scanCount == 1){
            vector<Mat> blockDigits0 = SplitBlock(numberWindow, points[0].x, points[0].y,
                                                  areasZ[0].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits0.begin(), blockDigits0.end() );

            vector<Mat> blockDigits1 = SplitBlock6(numberWindow, points[1].x, points[0].y,
                                                   areasZ[1].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits1.begin(), blockDigits1.end() );

            vector<Mat> blockDigits2 = SplitBlock5(numberWindow, points[2].x, points[0].y,
                                                  areasZ[2].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits2.begin(), blockDigits2.end() );
        } else if (scanCount == 2){
            for(auto it = begin(areasX); it < end(areasX); ++it, ++count) {
                vector<Mat> blockDigits = SplitBlock(numberWindow, points[count].x, points[0].y,
                                                     areasX[count].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
                if (blockDigits.size() != 4) return nullptr; //note

                digits.insert( digits.end(), blockDigits.begin(), blockDigits.end() );
            }
        } else {
            vector<Mat> blockDigits0 = SplitBlock(numberWindow, points[0].x, points[0].y,
                                                  areasY[0].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits0.begin(), blockDigits0.end() );

            vector<Mat> blockDigits1 = SplitBlock6(numberWindow, points[1].x, points[0].y,
                                                   areasY[1].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits1.begin(), blockDigits1.end() );

            vector<Mat> blockDigits2 = SplitBlock(numberWindow, points[2].x, points[0].y,
                                                  areasY[2].x, spaceBWDidits, paddingPoint.x, paddingPoint.y, digitRects);
            digits.insert( digits.end(), blockDigits2.begin(), blockDigits2.end() );
        }

//        if (blockDigits.size() != 4) return nullptr;

        shared_ptr<INeuralNetworkResultList> neuralNetworkResultDigits = _factory.lock()->CreateNeuralNetworkResultList();
        
        int minX = INT_MAX;
        int minY = INT_MAX;
        int maxX = -INT_MAX;
        int maxY = -INT_MAX;
        
        for(cv::Rect rect : digitRects) {
            minX = MIN(minX, rect.x);
            minY = MIN(minY, rect.y);
            maxX = MAX(maxX, rect.x + rect.width);
            maxY = MAX(maxY, rect.y + rect.height);
        }
        
        boundingRect = cv::Rect(minX, minY, maxX - minX, maxY - minY);
        
        Predict(digits, neuralNetworkResultDigits, _recognitionNeuralNetwork);
        
        if (ValidateNumber(neuralNetworkResultDigits)) {
            return neuralNetworkResultDigits;
        }
    }
    
    return nullptr;
}

bool CNumberRecognizer::ValidateNumber(const shared_ptr<INeuralNetworkResultList>& result)
{
    /// check probabilities
    float threshold; // 0.75
    if (scanCount == 1){
        threshold = 0.4;
    } else {
        __android_log_print(ANDROID_LOG_INFO, "THRESHOLD","Set threshold 0.7");
        threshold = 0.75;
    }
    const int maxDoubtfulCount = 1;
    
    int non = 0;
    
    for(INeuralNetworkResultList::ResultIterator it=result->Begin(); it != result->End(); ++it)
    {
        shared_ptr<INeuralNetworkResult> result = *it;
        
        if (result->GetMaxProbability() < threshold) {
            non++;
        }
    }
    
    if (non >= maxDoubtfulCount) {
        return false;
    }
//
    return CheckSum(result);
}

bool CNumberRecognizer::CheckSum(const shared_ptr<INeuralNetworkResultList>& result)
{
    vector<int> number = {};
    
    for(INeuralNetworkResultList::ResultIterator it=result->Begin(); it != result->End(); ++it)
    {
        shared_ptr<INeuralNetworkResult> result = *it;
        number.push_back(result->GetMaxIndex());
    }

//
//    if (number[0] != 5 && number[0] != 4 && number[0] != 2) {
//        return false;
//    }
//
//    if(number[0] == 2 && number[1] != 2) {
//        return false;
//    }
//
//    if (number[0] == 5 && (number[1] < 1 || number[1] > 5)) {
//        return false;
//    }
//
//    int k = 0;
//    if (number.size()%2 == 0) k = 1;
//
//    int sum = 0;
//    int tmp;
//
//    for(int i = 0; i < number.size(); i++)
//    {
//        tmp = number[i] * ((i+k)%2 + 1);
//        if(tmp > 9) tmp -= 9;
//            sum += tmp;
//            }
//
//    if(sum%10 != 0 ) return false;

    return true;
}
