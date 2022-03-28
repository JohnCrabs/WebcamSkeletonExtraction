import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):

        # MediaPipe Variables
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                     self.model_complexity,
                                     self.smooth_landmarks,
                                     self.enable_segmentation,
                                     self.smooth_segmentation,
                                     self.min_detection_confidence,
                                     self.min_tracking_confidence)
        self.results = None
        self.lmList = []

        # OpenPose Variables
        self._NET_PATH = "../dnn_model/graph_opt.pb"
        self._IMG_WIDTH = 368
        self._IMG_HEIGHT = 368
        self._IMG_THREAD = 0.2

        self._BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self._POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        self.net = cv2.dnn.readNetFromTensorflow(self._NET_PATH)

    def MediaPipe_findPose(self, img, draw=True):
        if img is not None and img.any():
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                               self.mpPose.POSE_CONNECTIONS)
        return img

    def MediaPipe_findPosition(self, img, draw=True):
        self.lmList = []
        if img is not None and img.any():
            if self.results is not None and self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    # print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def MediaPipe_findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def OpenPose_findPose(self, img):
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        self.net.setInput(cv2.dnn.blobFromImage(img, 1.0, (imgWidth, imgHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.net.forward()
        out = out[:, :19, :, :]

        assert(len(self._BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(self._BODY_PARTS)):
            # Slice heatmap of corresponding body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (imgWidth * point[0]) / out.shape[3]
            y = (imgHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self._IMG_THREAD else None)

        for pair in self._POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in self._BODY_PARTS)
            assert (partTo in self._BODY_PARTS)

            idFrom = self._BODY_PARTS[partFrom]
            idTo = self._BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = self.net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(img, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        return img


def MediaPipe_main():
    cap = cv2.VideoCapture('../../PoseVideos/pv_05.mp4')
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.MediaPipe_findPose(img)
        if img is not None and img.any():
            _ = detector.MediaPipe_findPosition(img, draw=True)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            scale_percent = 30  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Image", resized)
            cv2.waitKey(1)
        else:
            break


def OpenPose_main():
    cap = cv2.VideoCapture('../../PoseVideos/pv_05.mp4')
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.OpenPose_findPose(img)
        if img is not None and img.any():
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            scale_percent = 30  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Image", resized)
            cv2.waitKey(1)
        else:
            break


if __name__ == "__main__":
    # MediaPipe_main()
    OpenPose_main()
