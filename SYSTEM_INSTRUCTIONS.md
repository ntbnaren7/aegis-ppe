# GOAL.md

# AEGIS Project Goal

## Primary Objective

Develop AEGIS into a capable autonomous/semi-autonomous industrial safety and inspection rover powered by the Arduino UNO Q.

The immediate objective is to build and deploy a real-time camera perception system using the ROG Eye S USB webcam.

The system must eventually run on the actual Arduino UNO Q hardware rather than remaining a desktop-only proof of concept.

---

# Current Milestone

## AEGIS Vision V1

Build a real-time computer vision system capable of:

* Detecting industrial safety conditions
* Detecting PPE
* Detecting missing PPE
* Detecting humans
* Detecting fall events
* Detecting industrial safety objects
* Detecting ArUco markers
* Producing structured safety events
* Operating with useful real-time performance
* Running efficiently on Arduino UNO Q

---

# Required Safety Classes

The custom safety model must support all 14 classes:

0. Fall-Detected
1. Gloves
2. Goggles
3. Hardhat
4. Ladder
5. Mask
6. NO-Gloves
7. NO-Goggles
8. NO-Hardhat
9. NO-Mask
10. NO-Safety Vest
11. Person
12. Safety Cone
13. Safety Vest

The class IDs are:

```
0  = Fall-Detected
1  = Gloves
2  = Goggles
3  = Hardhat
4  = Ladder
5  = Mask
6  = NO-Gloves
7  = NO-Goggles
8  = NO-Hardhat
9  = NO-Mask
10 = NO-Safety Vest
11 = Person
12 = Safety Cone
13 = Safety Vest
```

All 14 classes are currently required.

---

# Current Dataset

The dataset is already prepared in YOLO object-detection format.

The structure is:

```
train/
├── images/
└── labels/

valid/
├── images/
└── labels/

test/
├── images/
└── labels/

data.yaml
```

Each image has a corresponding YOLO annotation file.

Image filenames may be random.

Random filenames are valid as long as:

```
images/example.jpg
```

matches:

```
labels/example.txt
```

The dataset must be validated before training.

---

# Navigation Vision System

ArUco markers are not part of the YOLO training dataset.

Use OpenCV's native ArUco detection.

ArUco is responsible for:

* Marker detection
* Marker ID extraction
* Position estimation
* Navigation context
* Route guidance
* Potential pose estimation

The high-level vision architecture is:

```
ROG Eye S Camera
        |
        +--> 14-Class Safety Detection
        |
        +--> OpenCV ArUco Detection
                 |
                 v
            Marker Information
                 |
                 v
            Navigation Context
```

The two perception systems may run from the same camera frame but should remain logically separate.

---

# Development Stack

Primary tools:

* Arduino UNO Q
* Arduino App Lab
* ROG Eye S webcam
* Python
* uv
* Ultralytics
* YOLO
* OpenCV

Additional deployment tools and runtimes must be selected based on verified Arduino UNO Q compatibility.

Do not assume a deployment format before verifying that it works with the target environment.

---

# Development Strategy

## Phase 1: Development Environment

Goals:

* Initialize the project with uv.
* Create a reproducible Python environment.
* Install required dependencies.
* Verify Ultralytics.
* Verify OpenCV.
* Verify PyTorch.
* Check GPU availability.
* Verify dataset paths.

Success criteria:

```
uv environment works
+
dataset is accessible
+
training dependencies are available
```

---

## Phase 2: Dataset Validation

Before training:

* Verify train paths.
* Verify validation paths.
* Verify test paths.
* Verify class count is 14.
* Verify data.yaml.
* Verify image-label matching.
* Check for corrupt images.
* Check YOLO label validity.
* Check class distribution.

Success criteria:

```
Dataset is structurally valid
+
labels match the expected classes
+
training can begin without data errors
```

---

## Phase 3: Baseline Training

Train a lightweight pretrained object-detection model.

The initial objective is not to create the final model.

The objective is to establish a measurable baseline.

Measure:

* Precision
* Recall
* mAP50
* mAP50-95
* Per-class performance
* False positives
* False negatives
* Training time
* Model size

The baseline must be recorded for future comparison.

---

## Phase 4: Validation and Testing

After training:

1. Evaluate validation performance.
2. Evaluate test performance.
3. Examine per-class metrics.
4. Identify weak classes.
5. Inspect false positives.
6. Inspect false negatives.
7. Record failure patterns.

Do not rely only on overall mAP.

Safety classes must be evaluated individually.

---

## Phase 5: Live ROG Eye S Testing

Run the trained model using the actual ROG Eye S webcam.

Test:

* Human detection
* PPE detection
* Missing PPE detection
* Fall detection
* Ladder detection
* Safety cone detection
* Different lighting
* Different distances
* Partial occlusion
* Motion
* Multiple people
* Industrial backgrounds

Record failure cases.

The actual camera environment is more important than only offline benchmark metrics.

---

## Phase 6: ArUco Integration

Integrate OpenCV ArUco detection into the live camera pipeline.

The system should simultaneously provide:

* Object detections
* PPE detections
* Missing PPE detections
* Safety violations
* Fall detections
* ArUco marker IDs
* ArUco position information where supported

Conceptually:

```
Camera Frame
    |
    +--> Safety Model
    |
    +--> ArUco Detector
             |
             v
      Combined Perception Output
```

---

## Phase 7: Safety Interpretation

Convert raw model detections into meaningful events.

Examples:

```
NO-Hardhat
-> PPE violation

NO-Safety Vest
-> PPE violation

NO-Gloves
-> PPE violation

NO-Goggles
-> PPE violation

Fall-Detected
-> Potential emergency event
```

Support states such as:

* COMPLIANT
* VIOLATION
* UNKNOWN

Do not automatically treat an undetected object as a violation.

Use temporal confirmation where appropriate.

Example:

```
Detection
    |
    v
Consecutive-frame confirmation
    |
    v
Safety event
```

---

## Phase 8: Performance Measurement

Measure:

* Camera FPS
* Inference FPS
* End-to-end FPS
* Inference latency
* End-to-end latency
* CPU usage
* RAM usage
* Model size

The complete system latency includes:

```
Camera Capture
+
Preprocessing
+
Model Inference
+
Postprocessing
+
Safety Interpretation
+
Event Processing
```

The final system must be evaluated using end-to-end performance.

---

## Phase 9: Model Optimization

Optimization happens only after a working baseline exists.

Possible experiments:

* Smaller model variants
* Different input resolutions
* Frame skipping
* FP16
* INT8
* Different compatible inference runtimes

Every optimization must be compared against the baseline.

Compare:

```
Accuracy
Latency
FPS
Memory
Model Size
```

The objective is:

> Maximum useful real-world performance within Arduino UNO Q resource limits.

Do not sacrifice critical safety detection accuracy merely to reduce model size.

---

## Phase 10: Arduino UNO Q Deployment

Deploy the selected model to the Arduino UNO Q.

Validate:

* Model loading
* Camera input
* Real-time inference
* End-to-end latency
* Memory stability
* CPU stability
* Runtime stability

The project is not considered successfully deployed until the model works on the actual Arduino UNO Q.

---

# Future AEGIS Integration

After the camera perception system is stable, integrate:

* Camera AI
* ArUco navigation
* Distance sensors
* IMU
* Environmental sensors
* Gas detection
* Temperature sensing
* Motor control

These systems will eventually contribute to:

```
Sensor Fusion
      |
      v
Decision Engine
      |
      v
Navigation
      |
      v
Rover Motor Control
```

---

# Future Safety Behavior

The eventual AEGIS system should:

* Detect safety violations.
* Detect potentially dangerous situations.
* Log safety events.
* Alert operators.
* Provide visual and structured context.
* Navigate safely.
* Avoid obstacles using dedicated ranging sensors.
* Continue operating when possible.
* Enter a safe state when critical subsystems fail.

---

# Definition of Success

AEGIS Vision is successful when the system:

1. Uses the ROG Eye S reliably.
2. Supports all 14 required safety classes.
3. Detects PPE and missing PPE.
4. Detects fall-related events.
5. Detects relevant industrial safety objects.
6. Detects ArUco markers.
7. Produces structured safety events.
8. Has measured validation and test performance.
9. Works on live camera footage.
10. Has measured real-world latency and FPS.
11. Is optimized based on actual measurements.
12. Runs successfully on the Arduino UNO Q.
13. Can later integrate with rover navigation and control.

The priority order is:

```
Correctness
    |
    v
Safety
    |
    v
Real-world Reliability
    |
    v
Deployability
    |
    v
Performance
    |
    v
Optimization
```

A model that trains successfully but cannot run reliably on the Arduino UNO Q is not considered a successful AEGIS solution.

---

# Immediate Next Goal

The immediate development objective is:

1. Set up the uv project.
2. Install the training dependencies.
3. Verify GPU availability.
4. Verify the YOLO dataset.
5. Train the first 14-class baseline.
6. Evaluate the model.
7. Test the model using the ROG Eye S.

Do not move to deployment optimization before the baseline model has been trained and evaluated.