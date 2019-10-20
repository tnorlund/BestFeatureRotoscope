#include <vector>

#ifndef CPP_MAIN_HPP_
#define CPP_MAIN_HPP_


// Initializing Video Stream
int loadVideo(
  char* filename,
  cv::VideoCapture* video
);
void getVideoProperties(
  int debug,
  cv::VideoCapture* video,
  double* FPS,
  double* MAX_TIME,
  double* start_time,
  int* MAX_FRAME,
  int* ROW,
  int* COL
);
int initVideoOutput(
  char* filename,
  cv::VideoWriter* output,
  int* ROW,
  int* COL,
  double* FPS
);

// Validating Input
int checkStartTime(
  int debug,
  char* time,
  double* start_time,
  double MAX_TIME
);
int checkEndTime(
  int debug,
  char* time,
  double* start_time,
  double* end_time,
  double MAX_TIME
);

// Reading frames from the video
int initImages(
  cv::VideoCapture* video,
  cv::Mat* image,
  cv::Mat* image_back,
  double back_time,
  double start_time,
  double FPS,
  int* current_frame
);
int getNextImage(
  int debug,
  cv::VideoCapture* video,
  cv::Mat* image,
  int* current_frame
);

// Rotoscope Functions
void downSample(
  cv::Mat* image,
  cv::Mat* image_ds,
  int factor,
  int COL,
  int ROW
);

/**
 * Calculates the centroid of the markers given.
 *
 * @param markers the markers previously calculated
 * @param center the (x,y) center of the markers given
 * @param factor the scalar used to scale the original image
 */
void GetCenter(
  std::vector<cv::Point2f> markers,
  int* center,
  int factor
);

void DrawFeatures_binary(
  cv::Mat* image,
  std::vector<cv::Point2f> markers,
  int factor
);
void DrawFeatures_markers(
  cv::Mat* image,
  std::vector<cv::Point2f> markers,
  int factor,
  int offset
);
void makeMask(
  cv::Mat* mask,
  int* center,
  int height,
  int width,
  int ROW,
  int COL
);
void waterShed_seg(
  cv::Mat* diff_image,
  cv::Mat* markers,
  int ROW,
  int COL
);
void matsnd(const cv::Mat& m, int dest);
cv::Mat matrcv(int src);

/**
 * Applies the average color found at the segmented area
 *
 * @param image[in] the original image
 * @param markers[in] the markers computed
 * @param out[out] the final image being written to
 * @param color the colors written to the image
 * @param maxIndex the largest index found in the image
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void colorPalette(
  cv::Mat* image,
  cv::Mat* markers,
  cv::Mat* out,
  int color[][4],
  int maxIndex,
  int ROW,
  int COL
);

/**
 * Merges an array for 
 *
 * @param arr the array to be merged
 * @param l the left index
 * @param r the right index
 */
void merge(double arr[], int l, int m, int r);

/**
 * Sorts an array of doubles using "merge sort"
 *
 * @param arr the array of doubles to sort
 * @param l the left index
 * @param r the right index
 */
void mergeSort(double arr[], int l, int r);

#endif  // CPP_MAIN_HPP_
