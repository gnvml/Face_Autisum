import cv2
import numpy
import sys
import numpy as np
import face_alignment
import threading
import tkinter

flag = False

def flag_run():
    global flag
    flag = not flag
    print("Flag:", flag)

def face_lmk():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    video_capture = cv2.VideoCapture(0)
    i = 0
    process_frame = True
    while True:
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size 
        # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if process_frame:
            try:
                if flag:
                    preds = fa.get_landmarks(frame)

                    for axis in preds[0]:
                        cv2.circle(frame, tuple(axis), 1, (0, 255, 0), 2)

                cv2.imshow('WebCam', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
        if flag:
            i += 1
            if i % 8 != 0:
                process_frame = False
            else:
                process_frame = True
        

    video_capture.release()
    cv2.destroyAllWindows()

def run():
    thr_run = threading.Thread(target = face_lmk)
    thr_run.start()
    print("start run !")


if __name__ == '__main__':
    window = tkinter.Tk()
    window.title("ECO4P")
    window.geometry("300x200")   

    btn_1 = tkinter.Button(window, text = "Run !", command = run)
    btn_1.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
    
    btn_2 = tkinter.Button(window, text = "Landmark!", command = flag_run)
    btn_2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

    
    window.mainloop()