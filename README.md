# Real-Time Facial Emotion Recognition

This project demonstrates the implementation of a real-time facial emotion recognition system using deep learning. The system utilizes the **DeepFace** library to detect and classify emotions from live webcam input and visualize the performance metrics (e.g., FPS, detection accuracy) in real-time.

### Project Thesis:
This project is part of a broader investigation into real-time emotion detection using deep learning. By integrating webcam-based emotion recognition with real-time feedback, the system provides a useful tool for emotion classification and performance tracking.

## Project Overview

- **Dataset**: Real-time webcam input capturing various facial expressions (happy, sad, angry, surprise, etc.).
- **Model Architecture**: DeepFace, pre-trained on facial emotion classification.
- **Task**: Real-time emotion detection and classification from live video, with performance tracking and visualization.

The project is implemented in a Python script (`main.py`) that uses webcam input to perform emotion detection and displays metrics on the fly.

## Requirements

To run the project and perform emotion recognition, the following libraries are required:

- Python 3.x
- OpenCV (`opencv-python`)
- DeepFace (`deepface`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)
- scikit-learn (`sklearn`)

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Real-Time-Facial-Emotion-Recognition.git
    cd Real-Time-Facial-Emotion-Recognition
    ```



## Running the Model

The model and emotion detection process are implemented in the `main.py` script. It follows these steps:

1. **Video Capture**: The webcam is activated to capture frames in real-time.
2. **Emotion Prediction**: Each frame is passed through the DeepFace model for emotion classification.
3. **Metrics Calculation**: Performance metrics such as FPS, detection accuracy, and emotion predictions are displayed in real-time.
4. **Result Saving**: Detected frames, performance graphs, and confusion matrix are saved to the `emotion_detection_results/` directory.

## Features

- Real-time emotion detection using webcam input.
- Emotion predictions for facial expressions such as happy, sad, angry, surprise, etc.
- Real-time visualization of performance metrics (FPS, detection accuracy).
- Saving of detection frames, metrics images, and performance graphs for analysis.
- Confusion matrix visualization for tracking model performance.

## Metrics

The following performance metrics are tracked and displayed:

- **FPS (Frames per Second)**: Real-time frame rate of the system.
- **Detection Accuracy**: The accuracy of emotion classification for each frame.
- **Successful Detections**: The number of frames with correctly predicted emotions.
- **Average Accuracy**: The average classification accuracy across all frames.

These metrics are shown live on the video feed and saved for later analysis.



## Contributions

Feel free to fork this repository and contribute by submitting issues, pull requests, or suggestions. Improvements to the model, training process, or feature additions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **DeepFace**: The model used for emotion detection is based on the pre-trained DeepFace library.
- **OpenCV**: Used for capturing webcam frames and displaying results in real-time.
- **CIFAR-10 Dataset**: Although this project works with real-time input, CIFAR-10 can be used for testing and evaluation purposes.
