import cv2

import imageio

cap = cv2.VideoCapture('data/test.mp4')

with open('results/test_pred_adam.txt') as f:
    pred = f.readlines()
with open('data/test.txt') as f:
    label = f.readlines()

i = 0

image = []

while (True):

    # Capture frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    cv2.putText(frame,
                'Prediction : ' + pred[i][:-1],
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    cv2.putText(frame,
                'Label : ' + label[i][:4],
                (50, 100),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    cv2.putText(frame,
                'Diff : ' + str(float(label[i][:4]) - float(pred[i][:-1]))[:4],
                (50, 150),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    i += 1
    # Display the resulting frame
    cv2.imshow('video', frame)

    if i<120:
        image.append(frame)
    if i==121:
        break

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()

imageio.mimsave('video_adam.gif', image, fps=5)