from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
def main():
    #print("hello,world")
    # read video frames
    input_video_path='input_videos/input_video.mp4'
    video_frames=read_video(input_video_path)
    # detecting players - persons
    player_tracker=PlayerTracker(model_path='yolov8x')
    player_detections=player_tracker.detect_frames(video_frames,
                                                   read_from_stub=False,
                                                   stub_path='tracker_stubs/player_detections.pkl'
                                                    )
    # detecting ball
    ball_tracker=BallTracker(model_path='models/yolo5_last.pt')
    ball_detections=ball_tracker.detect_frames(video_frames,
                                                   read_from_stub=False,
                                                   stub_path='tracker_stubs/ball_detections.pkl'
                                                    )
    # detecting court_lines_keypoints
    court_line_model_path='models/keypoints_model.pth'
    court_line_detector=CourtLineDetector(court_line_model_path)
    court_keypoints=court_line_detector.predict(video_frames[0])

    # choose player
    player_detections=player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    # Draw output
        # Draw Player Bounding boxes
    output_video_frames=player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames=ball_tracker.draw_bboxes(video_frames,ball_detections)
    output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)
    # Draw frame number
    for i,frame in enumerate(output_video_frames):
        cv2.putText(frame,f'Frame number: {i}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    save_video(output_video_frames,'output_videos/output_video.avi')
if __name__=='__main__':
    main()
