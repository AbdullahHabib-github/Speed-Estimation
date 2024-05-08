from ultralytics.solutions.speed_estimation import SpeedEstimator

def display_frames(self):
    """Display frame."""

    cv2.namedWindow('Ultralytics Speed Estimation', cv2.WINDOW_NORMAL)  # Make the window resizable
    cv2.resizeWindow('Ultralytics Speed Estimation', 1280, 960)  # Set desired width and height

    cv2.imshow("Ultralytics Speed Estimation", self.im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return


SpeedEstimator.display_frames = display_frames  


from ultralytics import YOLO
from ultralytics.solutions import speed_estimation

import cv2

model = YOLO("yolov8n.pt")
model.to('cuda')
names = model.model.names

videofile = "vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"

cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter(f"speed_estimation{videofile}.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(0, 900), (4000, 900)]

# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(names)