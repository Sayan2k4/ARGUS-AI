# Architecture

## Overview

ARGUS AI follows a modular pipeline where each stage is independent and can be extended without affecting the others.

```
Video Input
      │
      ▼
Vehicle Detection (YOLOv8)
      │
      ▼
Multi Object Tracking (ByteTrack)
      │
      ▼
Traffic Rule Analysis
      │
 ┌────┼───────────┬──────────┐
 │    │           │          │
 ▼    ▼           ▼          ▼
Helmet Triple   Wrong Way   OCR
Detection Riding Detection
      │
      ▼
Dashboard
```

## Components

### Detection

Detects vehicles and persons using YOLOv8.

### Tracking

Assigns persistent IDs to detected objects using ByteTrack.

### Rule Engine

Uses tracked objects to detect traffic violations.

### Dashboard

Displays processed events, violations and analytics.