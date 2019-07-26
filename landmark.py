import cv2
import numpy
import sys
import numpy as np
import face_alignment

if __name__ == '__main__':

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    i = 0
    while True:
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size 
        # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if process_this_frame:
            try:
                preds = fa.get_landmarks(frame)

                for axis in preds[0]:
                    cv2.circle(frame, tuple(axis), 1, (0, 255, 0), 2)

                cv2.imshow('WebCam', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
            
        i += 1
        if i % 7 != 0:
            process_this_frame = False
        else:
            process_this_frame = True

    video_capture.release()
    cv2.destroyAllWindows()