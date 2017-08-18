import math
import time


import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui


BOX = (1200, 250, 1670, 750)


def get_position(image):
    # Change the color-space
    image_blur = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    # Get the orange
    ret, thresh = cv2.threshold(image_blur, 200, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    if contours:
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(im.shape, np.uint8)
        mask = cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        moments = cv2.moments(mask)
        centre_of_mass = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
        return centre_of_mass


def reject_outliers(data):
    m = 1
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered


def moving(previous):
    positions = []
    prev = previous[69][0]

    for x, y in previous[-10:]:
        positions.append(x - prev)
        prev = x

    positions = reject_outliers(positions)

    total = 0
    for x in positions:
        total += math.sqrt(math.pow(x, 2))

    if total > 3:
        return True
    else:
        return False


def ready(previous):
    if 10 > previous[len(previous)-1][0] > -10:
        return True
    return False


def compensate(target, previous):
    positions = []
    prev = previous[54][0]

    for x, y in previous[-25:]:
        positions.append(x - prev)
        prev = x

    positions1 = reject_outliers(positions)
    if len(positions1) > 0:
        positions = positions1
    to_compensate = int(np.average(positions) * 24.5)
    print to_compensate

    return int(target[0] + to_compensate), target[1]


def shoot(pos):

    pyautogui.moveTo(1432, 858)  # moves mouse to X of 100, Y of 200.
    pyautogui.dragTo(1432 + (pos[0] / (1.6 - pos[1] / 315.0)), 600, 0.2, button='left')


def run_main():

    previous = []

    while True:
        # Grab the image
        image = np.array(ImageGrab.grab(bbox=BOX))
        target = get_position(image)
        if target:
            try:
                cv2.circle(image, target, 10, (0, 255, 0), -1)
            except:
                pass

            target = (target[0] - 230, target[1] - 302)

        length = len(previous)
        if length >= 80:
            if not moving(previous):
                shoot(target)
                pyautogui.click(1432, 925)
                previous = []
            else:
                if ready(previous):
                    shoot(compensate(target, previous))
                    pyautogui.click(1432, 925)
                    previous = []
                else:
                    previous = previous[-75:]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "{}".format(length), (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("MichalJordan.exe".format(length), image)
        previous.append(target)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    time.sleep(2)
    run_main()