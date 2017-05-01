[FileName, FilePath]=uigetfile('*');
inputVideo = VideoReader(strcat(FilePath, FileName));

inputVideoLength = inputVideo.Duration;

%Get Background
inputVideo.currentTime = 0;
backgroundFrame = readFrame(inputVideo);

disp(strcat('The input Video is ', num2str(inputVideoLength), 'seconds.'))
result = input('(1) Video\n(2) Frame\n');
numberOfFeatures = input('Number of Features: ');


if(result == 1)
    outputVideo = VideoWriter(strcat(FilePath(1:end-7), 'output/',...
    'roto',FileName(1:end-4)));
    open(outputVideo);
    startTime = input('Start Time (seconds)\n');
    inputVideo.CurrentTime = startTime;
    numberOfFrames = input('Number of Frames (number)\n');
    
    for time = 1:numberOfFrames
        disp(time);
        inputFrame = readFrame(inputVideo);
        outputImage = rotoscope(numberOfFeatures, backgroundFrame, inputFrame);
        writeVideo(outputVideo, outputImage);
    end
    close(outputVideo);
end

if (result == 2)
    time = input('Time (seconds)\n');
    inputVideo.CurrentTime = time;
    inputFrame = readFrame(inputVideo);
    outputImage = rotoscope(numberOfFeatures, backgroundFrame, inputFrame);
    imshow(outputImage)
    imwrite(outputImage, strcat(FilePath(1:end-7), 'output/',...
        'roto', FileName(1:end-4),'.png'), 'PNG');
end