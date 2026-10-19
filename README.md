# Multi-Sensor Triple Reconciliation Framework for Smart Retail Trolleys

A publication-grade smart shopping trolley system integrating **UHF RFID**, **YOLOv8 Computer Vision**, and **Load Cell Differential Mass Profiling** to eliminate inventory shrinkage, billing flickering, and theft during automated self-checkout.

Developed for research submission to **IEEE EmergIN 2027**.

---

## Key Architecture & Research Features

- **Triple Reconciliation Gate**: Enforces $N_{\text{RFID}} = N_{\text{Vision}} = N_{\text{Weight}}$ validation before unlocking checkout.
- **Computer Vision Pipeline**: Utilizes YOLOv8 with spatial IoU + centroid distance tracking, 3.0-second grace persistence buffering, and class label locking to prevent live video hallucinations and flickering.
- **EAS RFID Anti-Theft Exit Gate**: Electronic Article Surveillance (EAS) portal simulation that detects unpaid RFID tags at store exits and triggers instant security alarms.
- **Shopper Analytics**: Features dynamic 2D indoor aisle navigation mapping and real-time Eco-Health carbon footprint calculation ($\text{kg CO}_2\text{e}$).

---

## Items Detection

<img width="1373" height="631" alt="item detec1" src="https://github.com/user-attachments/assets/88cb7250-1545-4706-8045-962360e59a06" />

## Theft Detection
<img width="1377" height="632" alt="gate alrm" src="https://github.com/user-attachments/assets/f059f00f-4121-4cab-8b4d-3f0f54be164f" />

## Carbon Footprint
<img width="1372" height="627" alt="eco fp" src="https://github.com/user-attachments/assets/0c2837e0-c615-4185-950d-c717da0e06ae" />


## Adding/Removing
https://github.com/user-attachments/assets/e9fb19ed-6c80-433c-b2b9-751ace9e7560

## Nav,Footprint and Theft


https://github.com/user-attachments/assets/74ce7239-59ee-4259-91a5-f152acc6b432



## Project Structure

```text
SmartRetail_Research/
├── smart_trolley_mega_app.py   # Main Integrated POS Terminal Application
├── product_db.py               # SKU Metadata & Catalog Mapping
├── reconciliation_engine.py    # Triple Verification Logic
├── rfid_engine.py              # Hardware/Simulation RFID Engine
├── indoor_nav_engine.py        # A* Aisle Routing Engine
├── eco_health_engine.py        # Carbon Footprint & Energy Analytics
├── best.pt                     # Fine-tuned YOLO Weight Model
├── yolov8n.pt                  # Pre-trained COCO Base Weights
└── requirements.txt            # Environment Dependencies
```

---

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/DarKiller33/SmartRetail_Research.git
cd SmartRetail_Research
pip install -r requirements.txt
```

### 2. Execution

Run the primary POS terminal application from any terminal or IDE:

```bash
python smart_trolley_mega_app.py
```

---

## Interactive Keybindings

During live webcam demonstration, use the following hotkeys:

| Key | Function |
| :---: | :--- |
| **P** | Process Payment (Mark cart tags as PAID) |
| **E** | Pass Through RFID Exit Gate (Triggers Security Check) |
| **T** | Inject Unpaid Theft Tag into Cart (Simulates Shoplifting) |
| **N** | Toggle 2D Indoor Aisle Navigation Map Overlay |
| **H** | Toggle Eco-Health & Carbon Footprint Summary Overlay |
| **Q** | Quit Application |
