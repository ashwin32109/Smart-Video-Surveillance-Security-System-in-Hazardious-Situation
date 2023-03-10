#.................................Preprocess_The_Traindata.....................#
import cv2
import os

def video_to_frames(video, path_output_dir):
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

#video_to_frames('Test/Smoke.mp4', 'data/Train/smoke')
#video_to_frames('videos/Gun.mp4', 'data/clone/Gun2')
#video_to_frames('videos/Fight.mp4', 'data/clone/Fight')

