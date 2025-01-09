import cv2
from deepface import DeepFace
import numpy as np
from collections import deque
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def create_metrics_image(metrics_dict, size=(400, 300)):
    # Create a white image
    metrics_img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(metrics_img, "Performance Metrics", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2)
    
    # Add metrics
    y_position = 70
    for metric, value in metrics_dict.items():
        text = f"{metric}: {value}"
        cv2.putText(metrics_img, text, 
                    (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
        y_position += 40
    
    return metrics_img

def main():
    # Create output directory
    output_dir = "emotion_detection_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)
    
    emotion_queue = deque(maxlen=10)
    skip_frames = 5
    frame_count = 0
    
    # Performance metrics
    start_time = time.time()
    total_detections = 0
    successful_detections = 0
    fps_list = []
    accuracy_list = []
    emotion_predictions = []  # Add this to store predictions
    
    # For testing purposes, we'll use predictions as ground truth
    # In a real scenario, you would need actual ground truth labels
    actual_emotions = []  # Add this to store actual emotions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        fps = frame_count / (current_time - start_time)
        fps_list.append(fps)
        
        if frame_count % skip_frames == 0:
            try:
                total_detections += 1
                
                # Measure detection time
                detection_start = time.time()
                result = DeepFace.analyze(frame, actions=['emotion'], 
                                        enforce_detection=False, 
                                        detector_backend='opencv')
                detection_time = time.time() - detection_start
                
                emotion = result[0]['dominant_emotion']
                emotion_queue.append(emotion)
                emotion_predictions.append(emotion)  # Store prediction
                actual_emotions.append(emotion)  # For testing, using prediction as ground truth
                
                face_location = result[0]['region']
                x, y, w, h = face_location['x'], face_location['y'], face_location['w'], face_location['h']
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                accuracy = result[0]['emotion'][emotion]
                accuracy_list.append(accuracy)
                
                successful_detections += 1
                
            except Exception as e:
                print(f"Error in emotion detection: {str(e)}")
        
        if emotion_queue:
            smooth_emotion = max(set(emotion_queue), key=emotion_queue.count)
            
            # Add metrics to frame
            cv2.putText(frame, f"Emotion: {smooth_emotion}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Save frame with detection every 30 seconds
        if frame_count % (30 * int(fps)) == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Calculate and save metrics
            metrics = {
                "Average FPS": f"{np.mean(fps_list):.2f}",
                "Detection Rate": f"{(successful_detections/total_detections)*100:.2f}%",
                "Average Accuracy": f"{np.mean(accuracy_list):.2f}%",
                "Total Frames": str(frame_count),
                "Successful Detections": str(successful_detections),
                "Current Emotion": smooth_emotion
            }
            
            # Create and save metrics image
            metrics_img = create_metrics_image(metrics)
            metrics_filename = os.path.join(output_dir, f"metrics_{timestamp}.jpg")
            cv2.imwrite(metrics_filename, metrics_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save final metrics before quitting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_metrics = {
                "Total Runtime": f"{current_time - start_time:.2f} seconds",
                "Average FPS": f"{np.mean(fps_list):.2f}",
                "Detection Rate": f"{(successful_detections/total_detections)*100:.2f}%",
                "Average Accuracy": f"{np.mean(accuracy_list):.2f}%",
                "Total Frames": str(frame_count),
                "Successful Detections": str(successful_detections),
                "Most Common Emotion": smooth_emotion
            }
            
            metrics_img = create_metrics_image(final_metrics)
            final_metrics_filename = os.path.join(output_dir, f"final_metrics_{timestamp}.jpg")
            cv2.imwrite(final_metrics_filename, metrics_img)
            
            # Save visualization metrics before quitting
            if len(accuracy_list) > 0 and len(emotion_predictions) > 0:
                save_visualization_metrics(
                    accuracy_list,
                    emotion_predictions,
                    actual_emotions,
                    output_dir
                )
            break
    
    cap.release()
    cv2.destroyAllWindows()
def save_visualization_metrics(accuracy_list, emotion_predictions, actual_emotions, output_dir):
    """
    Save accuracy curve and confusion matrix as PNG files
    
    Parameters:
    accuracy_list: list of accuracy values over time
    emotion_predictions: list of predicted emotions
    actual_emotions: list of actual emotions (ground truth)
    output_dir: directory to save the images
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracy_list)), accuracy_list, '-b', linewidth=2)
    plt.title('Average Accuracy Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save accuracy curve
    accuracy_filename = os.path.join(output_dir, f'accuracy_curve_{timestamp}.png')
    plt.savefig(accuracy_filename)
    plt.close()
    
    # Create confusion matrix
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    conf_matrix = confusion_matrix(actual_emotions, emotion_predictions, labels=emotions)
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percentage, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                xticklabels=emotions,
                yticklabels=emotions)
    plt.title('Emotion Detection Confusion Matrix (%)')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Actual Emotion')
    plt.tight_layout()
    
    # Save confusion matrix
    matrix_filename = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(matrix_filename)
    plt.close()
    
    print(f"Saved visualizations to:\n{accuracy_filename}\n{matrix_filename}")


if __name__ == "__main__":
    main()