#  ARGUS AI

> **AI-Powered Traffic Intelligence Engine for Indian Roads**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Overview

ARGUS AI is a modular AI-powered traffic intelligence system designed to detect, track, and analyze vehicles in real time.

The long-term vision is to build an intelligent traffic monitoring platform capable of automatically identifying traffic-rule violations, generating analytics, and supporting smart-city infrastructure.

---

##  Current Features

-  Vehicle Detection
-  Person Detection
-  Multi Object Tracking

---

##  Upcoming Features

- Helmet Detection
- Triple Riding Detection
- Wrong Way Detection
- Red Light Violation Detection
- Number Plate OCR
- Analytics Dashboard

---

##  System Architecture

```text
Traffic Video
        │
        ▼
YOLOv8 Detection
        │
        ▼
ByteTrack Tracking
        │
        ▼
Violation Detection Engine
        │
 ┌──────┼────────┬────────┐
 │      │        │        │
 ▼      ▼        ▼        ▼
Helmet Triple WrongWay OCR
        │
        ▼
Dashboard
```

---

##  Tech Stack

- Python
- YOLOv8
- OpenCV
- ByteTrack
- NumPy

---

##  Project Structure

```text
src/
    detect.py
    track.py

config/
docs/
models/
outputs/
data/
```

---

##  Installation

```bash
git clone https://github.com/Sayan2k4/ARGUS-AI.git

cd ARGUS-AI

pip install -r requirements.txt
```

---

## ▶️ Run Vehicle Detection

```bash
python src/detect.py --video data/raw/sample1.mp4
```

---

## ▶️ Run Multi Object Tracking

```bash
python src/track.py --video data/raw/sample1.mp4
```

---

##  Development Progress

- [x] Vehicle Detection
- [x] Multi Object Tracking
- [ ] Helmet Detection
- [ ] Triple Riding Detection
- [ ] Wrong Way Detection
- [ ] Red Light Violation Detection
- [ ] Number Plate OCR
- [ ] Dashboard

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

##  Author

**Sayan Sarkar**

Building AI solutions for intelligent transportation and smart cities.