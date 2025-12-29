# DeepSORT Vehicle Tracking System

A real-time vehicle detection and tracking system using YOLOv11 + DeepSORT algorithm for military and civilian vehicle identification.

## Features

- **Multi-Object Tracking**: DeepSORT algorithm with Hungarian assignment and Kalman filtering
- **Custom Vehicle Detection**: Supports 6 vehicle classes including military vehicles
- **Real-time Processing**: GPU-accelerated inference with performance metrics
- **Accuracy Metrics**: MOTA, FP, FN, Identity Switches tracking
- **Local Data Storage**: Frame-by-frame data persistence with comprehensive analytics

## Vehicle Classes

| Class | Type | Threat Level | Priority |
|-------|------|--------------|----------|
| Car | CIVILIAN_CAR | LOW | 8 |
| Truck | TRANSPORT | MEDIUM | 7 |
| Bus | PUBLIC_TRANSPORT | LOW | 6 |
| Military Truck | MILITARY_VEHICLE | HIGH | 10 |
| Tank | MAIN_BATTLE_TANK | VERY_HIGH | 10 |
| Armored Tank | ARMORED_VEHICLE | VERY_HIGH | 10 |

## Installation

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install torch torchvision
pip install pandas numpy matplotlib
```

## Usage

```python
from main1 import process_video_with_enhanced_tracking

# Process video with custom model
output_path, report = process_video_with_enhanced_tracking(
    model_path="best.pt",
    video_path="input_video.mp4",
    max_frames=400,
    confidence=0.467
)
```

## DeepSORT Configuration

```python
tracker = DeepSort(
    max_age=50,           # Maximum frames to keep lost tracks
    n_init=3,             # Frames needed to confirm track
    max_cosine_distance=0.3,  # Feature matching threshold
    nn_budget=100,        # Maximum samples per class
    embedder="mobilenet", # Feature extraction model
    embedder_gpu=True,    # GPU acceleration
    bgr=True,            # BGR color format
    half=True            # FP16 precision
)
```

## Output Structure

```
VehicleTracking_Results_YYYYMMDD_HHMMSS/
├── processed_videos/     # Output videos with tracking
├── frame_data/          # JSON data per frame batch
├── analytics/           # CSV summaries and metrics
├── reports/            # Comprehensive analysis reports
├── visualizations/     # Performance plots
└── raw_tracking_data/  # Complete tracking data
```

## Accuracy Metrics

- **MOTA**: Multiple Object Tracking Accuracy
- **FP**: False Positives count
- **FN**: False Negatives count  
- **IDS**: Identity Switches count
- **Processing Time**: Per-frame inference time

## Key Components

### VehicleTracker Class
- Initializes YOLO model and DeepSORT tracker
- Configures vehicle-specific parameters
- Sets up local storage structure

### Frame Processing
- YOLO detection with confidence filtering
- DeepSORT tracking with Hungarian algorithm
- Kalman filter for motion prediction
- Accuracy metrics calculation

### Visualization
- Real-time bounding boxes with threat levels
- Track trails for high-priority vehicles
- Performance overlay with metrics
- Color-coded threat level indicators

## Performance

- **GPU Acceleration**: CUDA support for both YOLO and DeepSORT
- **Real-time Processing**: Optimized for live video streams
- **Memory Efficient**: Configurable history buffers
- **Batch Processing**: Interval-based data saving

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Custom trained YOLO model (`best.pt`)
- Input video file

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Contact

For questions or issues, please open a GitHub issue.
