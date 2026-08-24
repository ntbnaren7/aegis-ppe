# AEGIS-PPE (Personal Protective Equipment Monitoring)

## Project Overview
AEGIS-PPE is a computer vision project utilizing YOLO-based object detection to monitor compliance with safety gear requirements in real-time. The model detects multiple critical safety classes, including Fall-Detected, Hardhat, Safety Vest, Mask, and Person.

## Hardware Deployment & Benchmarking
The primary deployment target for this project is the Arduino UNO Q, which features the Qualcomm QRB2210 (Kryo-V2 Cortex-A53) and a Hexagon DSP.

During Phase 10 hardware profiling, the pipeline was quantized to INT8 and evaluated at multiple resolutions (640, 512, 416, 320) using the ONNX Runtime CPU Execution Provider. 
- 416x416 was determined to be the optimal CPU compromise, retaining 0.7388 mAP50 while achieving 3.11 FPS.
- CPU inference falls short of the provisional 8 FPS system requirement.
- The 320x320 resolution suffers from severe degradation in accuracy (-0.077 mAP50 drop and high false positive rates) and was rejected.

## Hardware Acceleration Limitations
An exhaustive hardware transport audit of the Arduino UNO Q Debian 13 (Trixie) image revealed critical blockers for native QNN/Hexagon offloading:
1. FastRPC kernel transport is present, but lacks the necessary Contiguous Memory Allocator (CMA) pool (no reserved DMA memory for FASTRPC), preventing the kernel from allocating tensor buffers.
2. The /dev/fastrpc-adsp device node is locked to the root user, blocking unprivileged applications.
3. The Qualcomm Neural Network (QNN) SDK and HTP runtime binaries are entirely absent from the root filesystem.
4. No mediating systemd daemons exist to handle DSP abstraction.

## Final Deployment Strategy
Due to the OS-level restrictions on direct DSP access in the Arduino Debian environment, native ONNX Runtime QNN acceleration is not viable. 

To bypass these limitations and successfully leverage hardware acceleration, the YOLO model must be exported and imported into Edge Impulse. The final compiled inference engine will be built and deployed specifically for the Arduino UNO Q via the official Edge Impulse toolchain.
