import cv2
import json
import time
import os
from model import load_model, detect_objects
from utils import create_json_output, crop_and_save

def process_frame(frame, model, labels, frame_count, output_json):
    start_time = time.time()
    
    # Detect objects and sub-objects
    detections = detect_objects(model, frame, labels)
    
    # Generate JSON output
    json_data = create_json_output(detections)
    output_json.append(json_data)
    
    # Save sub-object images
    crop_and_save(detections, frame, frame_count)
    
    # Benchmark FPS
    end_time = time.time()
    print(f"Frame {frame_count}: {1 / (end_time - start_time):.2f} FPS")

def main(video_path, output_json_path='outputs/detections.json', display_frames=True, skip_frames=0):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Load model
    model, labels = load_model()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Error handling if video path is invalid
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    output_json = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame for faster processing (Optional: Resize to 640x480 or any smaller resolution)
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Skip frames for performance optimization
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue
        
        frame_count += 1
        
        # Process the frame
        process_frame(frame_resized, model, labels, frame_count, output_json)

        # Show frame (Optional)
        if display_frames:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save JSON output
    with open(output_json_path, 'w') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Processing complete. Outputs saved to {output_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help="Path to the input video file")
    parser.add_argument('--output_json', default='outputs/detections.json', help="Path to save the JSON output")
    parser.add_argument('--display', action='store_true', help="Display video frames while processing")
    parser.add_argument('--skip_frames', type=int, default=0, help="Number of frames to skip for performance optimization")
    args = parser.parse_args()
    
    main(args.video, output_json_path=args.output_json, display_frames=args.display, skip_frames=args.skip_frames)
