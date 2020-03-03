from cv2 import cv2

cap = cv2.VideoCapture('../data/video/US_2020022502_m_29_170_72_008.mp4')
i = 0
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('../data/video_image/US_2020022502_m_29_170_72_008.mp4/US_2020022502_m_29_170_72_008_'+str(i)+'.png', frame)
        i += 1