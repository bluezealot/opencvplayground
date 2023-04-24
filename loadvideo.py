import cv2

def playvideo(path):
    cap = cv2.VideoCapture('samples/My Movie.mov')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', cv2.resize(frame, (0, 0), fx=0.8, fy=0.8))
            # Press any key to exit
            if cv2.waitKey(10) == ord('q'):
                break
        # Break the loop
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def captureimg(path):
    cap = cv2.VideoCapture('samples/My Movie.mov')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    duration = 30
    frame_count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if frame_count % duration == 0:
                # Output the resulting frame
                cv2.imwrite('generate/frame'+ '{:n}.jpg'.format(frame_count/duration).rjust(9, '0'), frame)
            frame_count += 1
        # Break the loop
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    captureimg('samples/My Movie.mov')