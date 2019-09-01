

#ifndef _ROTOSCOPE_MAIN_H_
#define _ROTOSCOPE_MAIN_H_


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
void GetCenter(
  std::vector<cv::Point2f> corners,
  int* center,
  int factor
);
void DrawFeatures_binary(
  cv::Mat* image,
  std::vector<cv::Point2f> corners,
  int factor
);
void DrawFeatures_markers(
  cv::Mat* image,
  std::vector<cv::Point2f> corners,
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
void colorPalette(
  cv::Mat* image,
  cv::Mat* markers,
  cv::Mat* out,
  int color[][4],
  int maxIndex,
  int ROW,
  int COL
);


#endif // _ROTOSCOPE_MAIN_H_
