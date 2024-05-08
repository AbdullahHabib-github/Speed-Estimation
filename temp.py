from ultralytics.solutions.speed_estimation import SpeedEstimator
from time import time
from ultralytics.utils.plotting import Annotator
import numpy as np


def calculate_speed(self, trk_id, track,box):
    """
    Calculation of object speed.

    Args:
        trk_id (int): object track id.
        track (list): tracking history for tracks path drawing
    """

    if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
        return
    if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
        direction = "known"

    elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
        direction = "known"

    else:
        direction = "unknown"

    if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
        self.trk_idslist.append(trk_id)

        time_difference = time() - self.trk_previous_times[trk_id]
        if time_difference > 0:
            dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
            speed = dist_difference / time_difference
            self.dist_data[trk_id] = speed
            print(f"trk_id: {trk_id}, ",f"speed: {speed}, ",f"box: {box} ")

    self.trk_previous_times[trk_id] = time()
    self.trk_previous_points[trk_id] = track[-1]


SpeedEstimator.calculate_speed = calculate_speed


def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
    """
    Calculate object based on tracking data.

    Args:
        im0 (nd array): Image
        tracks (list): List of tracks obtained from the object tracking process.
        region_color (tuple): Color to use when drawing regions.
    """
    self.im0 = im0
    if tracks[0].boxes.id is None:
        if self.view_img and self.env_check:
            self.display_frames()
        return im0
    self.extract_tracks(tracks)

    self.annotator = Annotator(self.im0, line_width=2)
    self.annotator.draw_region(reg_pts=self.reg_pts, color=region_color, thickness=self.region_thickness)

    for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
        track = self.store_track_info(trk_id, box)

        if trk_id not in self.trk_previous_times:
            self.trk_previous_times[trk_id] = 0

        self.plot_box_and_track(trk_id, box, cls, track)
        self.calculate_speed(trk_id, track,box)

    if self.view_img and self.env_check:
        self.display_frames()

    return im0

SpeedEstimator.estimate_speed = estimate_speed



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
                   view_img=False)

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, verbose = False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    
    
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
