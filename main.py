import cv2
import argparse
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
import supervision as sv
import numpy as np
import imutils
points = []
def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    
        
    return polygon.contains(centroid)
def handle_left_click(event, x, y, flags, points):
   
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame        
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")
   

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

   
    
   
    while True:
        ret, frame = cap.read()
        frame=imutils.resize(frame, width=800)
        
        results = model(frame, agnostic_nms=True)[0]
            
        for result in results:
        
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    x2 = int(x2)
                    y1 = int(y1)
                    y2 = int(y2)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, centroid, 5, (255,0,0), -1)  
                  
             
        detections = sv.Detections.from_yolov8(results)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        frame = draw_polygon(frame, points)
        if isInside(points, centroid)==False:
            cv2.putText(frame, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return isInside(points, centroid)
       
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        cv2.imshow("yolov8", frame)
        cv2.setMouseCallback('yolov8', handle_left_click, points)        

if __name__ == "__main__":
    main()