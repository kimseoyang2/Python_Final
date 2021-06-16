# -*- coding: utf-8 -*-
"""
@author: syeminPark
"""
from os import stat
from google.protobuf.text_format import PrintField
import UdpComms as U
import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import csv
# 라이다 pip install pyserial
import serial
from serial import Serial
from source import head_pose_points


########################################################################################################
# 내가 만든 함수

# 라이다 csv에 입력
def addDis(val):
    global total
    total += val
    # print(total)


def timer(f):
    global testCounter
    global state
    testCounter += 1
    if(testCounter > f):
        testCounter = 0
        print("state=", state)
        state += 1


def decideAI(val1, val2):
    if val1 > val2:
        return "OVER"
    else:
        return "UNDER"


def decideLidar(val1, val2):
    if val1 > val2:
        return "CLOSE"
    else:
        return "FAR"


def sockSend(sent, val):
    sock.SendData(val)
    print(val)
    if(sent):
        return False
    else:
        return True


def reset():
    global counter
    global closeCounter
    global farCounter
    global total
    global resetTimer
    global addTimer
    global dis

    counter = 0
    closeCounter = 1
    farCounter = 1
    total = 0
    resetTimer = 0
    addTimer = 0
    dis = 0
    isSent==False


def getTFminiData():
    if ser.is_open == False:
        ser.open()

    while True:
        if ser.is_open == False:
            ser.open()

        # time.sleep(0.1)
        count = ser.in_waiting
        if count > 8:
            recv = ser.read(9)
            ser.reset_input_buffer()

            if recv[0] == 0x59 and recv[1] == 0x59:  # python3
                distance = recv[2] + recv[3] * 256
                strength = recv[4] + recv[5] * 256

                if distance < maxDist:
                    if distance < line:
                        global closeCounter
                        closeCounter += 1
                        addDis(distance)
                    else:
                        global farCounter
                        farCounter += 1
                        addDis(distance)

            break


def getAIData():
    while True:
        global img
        ret, img = cap.read()
        if ret == True:
            faces = find_faces(img, face_model)
            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array(
                    [
                        marks[30],  # Nose tip
                        marks[8],  # Chin
                        marks[36],  # Left eye left corner
                        marks[45],  # Right eye right corne
                        marks[48],  # Left Mouth corner
                        marks[54],  # Right mouth corner
                    ],
                    dtype="double",
                )
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                global success
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_UPNP,
                )

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 1000.0)]),
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs,
                )

                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]),
                      int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(
                    img, rotation_vector, translation_vector, camera_matrix
                )

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1 / m)))
                except:
                    ang2 = 90

                computeHead(ang1, ang2, p1, x1)

            break


def computeDis():
    global dis
    global total
    dis = total / (closeCounter + farCounter)


def computeHead(angY, angX, p1, x1):
    if angY >= move:
        cv2.putText(img, "Head down", (30, 30), font, 2, (255, 255, 128), 3)
        add()
    elif angY <= -move:
        cv2.putText(img, "Head up", (30, 30), font, 2, (255, 255, 128), 3)
        add()
    if angX >= move:
        cv2.putText(img, "Head right", (90, 30), font, 2, (255, 255, 128), 3)
        add()

    elif angX <= -move:
        cv2.putText(img, "Head left", (90, 30), font, 2, (255, 255, 128), 3)
        add()

    if angX > -move and angX < move:
        global resetTimer
        global count
        global addTimer

        resetTimer += 1
        if resetTimer > 10:
            count = True
            resetTimer = 0
            addTimer = 0

    cv2.putText(img, str(angY), tuple(p1), font, 2, (128, 255, 255), 3)
    cv2.putText(img, str(angX), tuple(x1), font, 2, (255, 255, 128), 3)


def add():
    global count
    global counter
    global resetTimer
    global addTimer

    addTimer += 1
    if count:
        if addTimer > 5:
            counter += 1
            count = False
            print("Counter:", counter)
    resetTimer = 0


def saveCSV():
    global peopleNum
    global dis
    text = str(dis)
    f = 0
    f = open("csv/file.csv"
             "",
             "a",
             encoding="utf-8",
             newline="",
             )
    wr = csv.writer(f)
    if peopleNum == 0:
        wr.writerow(["Num", "Counter", "Distance"])
        wr.writerow([peopleNum, counter, text])

    else:
        wr.writerow([peopleNum, counter, text])
    f.close()
    peopleNum += 1


def receiveUnity():
    data = sock.ReadReceivedData()
    if data != None:
        global state
        # if NEW data has been received since last ReadReceivedData function call
        print("from Unity", data)  # print new received data
        state = int(data)


########################################################################################################
# 본문

# 파리미터
line = 50  # 멀고 가깝고 기준
maxDist = 130  # 센서 벗어나는 거리
limit = 2  # AI 움직임 개수 파라미터
move = 40  # 머리 움직임량 기준
ser = serial.Serial("COM7", 115200)
# ser = serial.Serial("//dev/cu.usbserial-1420", 115200)
state = 0

# 전역변수들
dis = 0
peopleNum = 0
isSent =False
counter = 0
count = True
resetTimer = 0
closeCounter = 1
send = True
farCounter = 1
total = 0
timeCounter = 0
addTimer = 0
testCounter = 0

# 유디피 통신 선언
sock = U.UdpComms(
    udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True


)
face_model = get_face_detector()
landmark_model = get_landmark_model()

cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX


# 3D model points.
model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        # Right eye right corne
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ]
)

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)


while True:

    receiveUnity()
    if state == 0:
        reset()
        
        testCounter += 1
      

    elif state == 1:
        getTFminiData()
        getAIData()
     

    elif state == 2:
        getAIData()

        
        computeDis()
        resultLidar = decideLidar(closeCounter, farCounter)
        isSent = sockSend(isSent, resultLidar)

       

    elif state == 3:
       
        resultAI = decideAI(counter, limit)
        isSent = sockSend(isSent, resultAI)

       

    elif state == 4:
        saveCSV()
        print("Totalcounter=", counter, "AverageDistance",
              dis, "loopNum=", peopleNum)
        state = 0

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
