# TCS Technical Interview Preparation Guide

This document contains simplified, easy-to-learn answers to key technical questions about the **SketchCalibur** project. Definitions of technical terms are provided in brackets to help you answer questions confidently without inviting deep, complex follow-ups.

---

## 1. Python Libraries & ML Production Stack

### Which Python libraries did you use in the ML Backend?
* **FastAPI**: A web framework (software for building APIs) used to create the server endpoints. It is selected because it is extremely fast and natively handles asynchronous code (doing multiple tasks without blocking the server).
* **NumPy**: A math library used for handling images as multi-dimensional grids of numbers (matrices).
* **OpenCV (opencv-python-headless)**: A computer vision library used for image operations like resizing, thresholding (converting grayscale to pure black/white), and finding contours (outlines of shapes). We used the `headless` version because serverless runtimes don't have graphical user interfaces (GUIs).
* **Pillow (PIL)**: A library used for basic image opening, saving, and cropping.
* **ONNX Runtime (onnxruntime)**: A lightweight execution engine used to run our trained PyTorch models on the CPU.
* **SymPy**: A symbolic mathematics library used to parse, simplify, and solve handwritten math equations.
* **google-generativeai**: The official client library used to communicate with the Gemini API.

### Why did you use this stack instead of heavy frameworks (like PyTorch or TensorFlow) in production?
1. **Size Limits**: Production servers like Vercel have a strict **500MB unzipped storage limit**. A full installation of PyTorch is over **1.5GB**, which would fail to deploy. ONNX Runtime is under **30MB**.
2. **Memory (RAM) Footprint**: Free hosting platforms (like Render or Vercel) limit RAM to **512MB - 1GB**. Loading full PyTorch models causes Out-Of-Memory (OOM) crashes. ONNX Runtime uses minimal RAM.
3. **Speed (Cold Starts)**: Loading PyTorch takes several seconds to initiate, causing high latency (latency: request-response delay) on serverless platforms. ONNX model loading is near-instantaneous.
4. **No local OCR engine**: Instead of running a heavy local OCR (Optical Character Recognition: converting image text to editable text) library like Tesseract or EasyOCR, we used cloud APIs (a HuggingFace space running PaddleOCR-VL and the Gemini API). This keeps our server's size and memory footprint virtually zero.

---

## 2. WebSockets vs. HTTP

### Why did you use WebSockets instead of HTTP for the collaborative canvas?
* **HTTP** is a unidirectional (one-way) protocol. The client must send a request, and the server returns a response. To get real-time updates of other users drawing, clients would have to use **polling** (repeatedly asking the server "Is there new data?" every 100ms). This creates massive HTTP header overhead and lags the canvas.
* **WebSockets** provide a **persistent, full-duplex (two-way) connection** over a single TCP (Transmission Control Protocol: connection-oriented delivery network) channel. Once connected, client and server can send data (like drawing coordinates) back and forth instantly with almost zero header overhead. This drops latency to under 15ms.

### What are the alternative strategies?
* **WebRTC (Web Real-Time Communication)**: A peer-to-peer (direct communication between browsers without a central server) protocol.
  * *Why it is better*: It bypasses the server completely for coordinate exchange, reducing server load and lowering latency. It can use UDP (User Datagram Protocol: a fast, connectionless network protocol that doesn't wait for packet confirmations), which is even faster than TCP.
  * *Why we didn't use it*: It is significantly more complex to implement and requires setup of STUN/TURN servers to bypass user firewalls.
* **SSE (Server-Sent Events)**: A one-way channel where the server pushes updates to the client. It is lighter than WebSockets but doesn't allow the client to send data back over the same connection (clients would still have to use HTTP POST to send their drawings).

---

## 3. Conflict-Free State Reconciliation

### What do you mean by Conflict-Free State Reconciliation?
When multiple users draw or edit the same canvas at the same time, their changes can conflict (e.g., User A moves a circle while User B deletes it).
**Conflict-Free State Reconciliation** is the strategy of merging these edits automatically so that every user's screen converges (ends up looking identical) without using a locking mechanism or throwing errors.

### How did you achieve it in this project?
1. **Independent Elements**: In Excalidraw, the canvas is not a single image. It is a list of independent shape elements (like lines, circles, and text). Each element has a globally unique ID (`uuid`), a version number (integer starting at 1), and a timestamp.
2. **Granular Updates**: When a user draws or edits a shape, we only transmit that specific updated element over WebSockets, not the whole canvas.
3. **Last-Write-Wins (LWW) Rule**: When a client receives an element update, it checks the incoming element's version and timestamp.
   * If the incoming version is higher than the local version, the local shape is updated.
   * If it is lower or equal, it is safely discarded.
4. This ensures conflict-free synchronization without complex merging algorithms.

---

## 4. Pix2Pix GAN Architecture & Training

### How does the Pix2Pix GAN Architecture work?
A GAN (Generative Adversarial Network) consists of two neural networks playing a game:
1. **Generator**: Takes a messy, wobbly hand-drawn sketch and tries to output a clean, straight line-art drawing.
2. **Discriminator**: Takes pairs of images—either (messy sketch, real clean drawing) or (messy sketch, generator's clean drawing)—and classifies them as "real" or "fake".
*During training, they compete: the Generator gets better at drawing clean lines to fool the Discriminator, while the Discriminator gets better at catching flaws.*

### Where did you get the dataset from?
* We used **Google's QuickDraw Dataset**, which is a public collection of millions of clean vector drawings drawn by humans.
* We streamed and packed these drawings into a compressed `.npz` (NumPy compressed archive) file format.

### How did you generate the "messy" sketches for training?
Since GANs require paired images, we wrote a **synthetic corruption pipeline** that took clean drawings and made them messy:
1. **Elastic Deformations**: Used wobbly grid transformations to make straight lines look shaky (simulating a human hand).
2. **Line Breaks**: Added random white shapes/lines to simulate eraser marks or incomplete strokes.
3. **Ink Bleeding**: Applied Gaussian blurs to simulate ink spreading on paper.
4. **Motion Blur & Noise**: Added pixel noise to simulate cheap stylus inputs or scanning artifacts.

### What were the hyperparameters and fine-tuning configurations?
* **Generator Model**: U-Net 256 (a network that downsamples an image to extract features, then upsamples it, using skip connections to preserve fine details like sharp corners).
* **Discriminator Model**: PatchGAN 70x70 (classifies 70x70 pixel patches as real/fake to enforce sharp local details).
* **Epochs**: 20 (outline generation converges very fast compared to color photos).
* **Batch Size**: 128 (saturates Kaggle's NVIDIA T4 GPU).
* **Learning Rate**: 0.0002 with Adam Optimizer ($\beta_1=0.5, \beta_2=0.999$, standard for GANs), decaying linearly to 0 in the second half of training.
* **Loss Function**: GAN Loss (MSELoss) + $\lambda \times$ L1 Loss (with $\lambda=100.0$). L1 loss enforces that the output looks structurally identical to the input sketch.
* **Training Accelerators**: Auto Mixed Precision (AMP FP16) to speed up training by 2x-3x, and in-memory RAM caching to avoid slow disk read bottlenecks.

---

## 5. Resume Points Q&A

### Resume Bullet 1:
> *"Engineered a real-time collaborative canvas supporting 140+ users with live cursor sync and conflict-free state reconciliation via optimized WebSocket event handling."*

#### Q: How did you test that the server can support 140+ concurrent users?
**A:** We wrote a custom load-testing script in TypeScript. The script generates 100+ simulated users, signs authenticated JWT (JSON Web Tokens: a secure way to transmit user session data) tokens for them, connects them to our room over WebSockets, and continuously transmits cursor coordinate packages at 20Hz (20 times per second) for 10 seconds. We monitored CPU/memory and broadcast latency.

#### Q: What optimizations did you perform on WebSocket event handling to support this load?
**A:** We did three key optimizations:
1. **Active Drawer Cap**: We capped active drawers per room at 10 users, switching the remaining users to view-only listeners. This reduced broadcast operations by 99%.
2. **Client-Side Message Batching**: Cursors are batched. Instead of sending a socket message for every single pixel moved, coordinates are grouped over a 100ms window and sent as a single packet.
3. **Redis Pub/Sub Horizontal Scaling**: Distributed WebSocket connections across multiple application instances using Redis Pub/Sub (Publish/Subscribe: a message passing pattern) to broadcast messages across server instances.

---

### Resume Bullet 2:
> *"Achieved session retention through secured room sharing; integrated AI-driven diagram generation from natural language prompts using Gemini LLM."*

#### Q: What is "session retention" and how was it secured?
**A:** Session retention means users can refresh the browser or return later and still find their room and drawings intact. We achieved this by persisting room states in MongoDB. Access to rooms was secured by signing JWT tokens during sharing. Only users with the signed JWT could connect to that room ID.

#### Q: How does the AI-driven diagram generation work?
**A:** When a user types a prompt (e.g., "draw a flowchart of a login system"), we send that text to the Gemini API. We prompt Gemini to generate the diagram elements strictly in **Excalidraw JSON format** (specifying positions, labels, shapes, and arrows). Our frontend then parses this JSON and injects the shapes directly onto the canvas.

---

### Resume Bullet 3:
> *"Built a multi-modal AI pipeline integrating PaddleOCR-VL for handwriting transcription, Pix2Pix GAN for sketch enhancement and a Math solver using YOLOv8 & SymPy – deployed across free-tier microservices"*

#### Q: Why did you choose PaddleOCR-VL for handwriting transcription?
**A:** Handwriting is highly irregular and difficult for classical OCR engines (like Tesseract) to read. PaddleOCR-VL (Vision-Language) is a deep-learning OCR model pre-trained specifically on multilingual layout analysis and handwritten texts, making it highly accurate for notes. We deployed it on a free HuggingFace Space to offload CPU and memory costs.

#### Q: How does the Math Solver pipeline work?
**A:** 
1. The user selects a handwritten equation on the canvas.
2. We crop the image and send it to our backend.
3. We run OCR to transcribe the image to a text string (using a 3-tier fallback pipeline: Roboflow API -> Gemini -> EasyOCR).
4. Once we have the equation text string (e.g. `x^2 - 4 = 0`), we pass it to **SymPy** (a symbolic math solver) which calculates the step-by-step solution.
5. The solved steps are returned as a styled text element on the canvas.
