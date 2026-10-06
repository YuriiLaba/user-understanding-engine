import cv2
import numpy as np
import os

def calculate_frame_difference(frame1, frame2):
    """Calculate the difference between two frames using structural similarity"""
    if frame1 is None or frame2 is None:
        return float('inf')
    
    # Convert to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # For screen recordings, use absolute difference sum (more sensitive to UI changes)
    # Also helps detect cursor movement, text changes, etc.
    diff = cv2.absdiff(gray1, gray2)
    
    # Count pixels that changed significantly (threshold out minor compression artifacts)
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    changed_pixels = np.sum(thresh > 0)
    
    # Return percentage of changed pixels
    total_pixels = thresh.size
    change_percentage = (changed_pixels / total_pixels) * 100
    
    return change_percentage

def save_stable_frames(video_paths, output_dir='output_frames', 
                       diff_threshold=0.5, stability_duration=1.0):
    """
    Save frames that are different from previous saved frame and stable for at least 1 second
    Optimized for screen recordings
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        diff_threshold: Percentage threshold (0-100) to consider frames different
        stability_duration: Minimum time (seconds) a frame must be stable before saving
    """
    
    if os.path.exists(output_dir):
        print(f"[INFO] Folder '{output_dir}' already exists. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    last_saved_frame = None
    candidate_frame = None
    candidate_frame_idx = None
    stable_count = 0
    saved_count = 0
    frame_idx = 0
    total_frames = 0


    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_per_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stability_frame_count = int(fps * stability_duration)
        total_frames += total_frames_per_video
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Stability requirement: {stability_frame_count} frames ({stability_duration}s)")
        print(f"Difference threshold: {diff_threshold}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Check if current frame is different from last saved frame
            diff_from_saved = calculate_frame_difference(frame, last_saved_frame)
            is_different_from_saved = diff_from_saved > diff_threshold
            
            if is_different_from_saved:
                # Check if this is similar to our current candidate
                if candidate_frame is not None:
                    diff_from_candidate = calculate_frame_difference(frame, candidate_frame)
                    is_similar_to_candidate = diff_from_candidate <= diff_threshold
                    
                    if is_similar_to_candidate:
                        # Frame is stable, increment counter
                        stable_count += 1
                        
                        # Check if we've reached stability threshold
                        if stable_count >= stability_frame_count:
                            # Save the candidate frame
                            output_path = os.path.join(output_dir, f'frame_{saved_count:04d}_idx_{candidate_frame_idx}.jpg')
                            cv2.imwrite(output_path, candidate_frame)
                            print(f"Saved: {output_path} (stable for {stable_count} frames)")
                            
                            last_saved_frame = candidate_frame.copy()
                            candidate_frame = None
                            stable_count = 0
                            saved_count += 1
                    else:
                        # Frame changed, reset candidate
                        candidate_frame = frame.copy()
                        candidate_frame_idx = frame_idx
                        stable_count = 0
                else:
                    # Start tracking a new candidate frame
                    candidate_frame = frame.copy()
                    candidate_frame_idx = frame_idx
                    stable_count = 0
            else:
                # Frame is similar to last saved, reset candidate
                candidate_frame = None
                stable_count = 0
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        print(f"\nCompleted! Saved {saved_count} stable frames to '{output_dir}'")