# README.md

## Project Title: **Player Tracking in Football Video Using YOLOv8 + IOU Matching**

---

### ✅ How to Set Up and Run the Code

1. Place the following files in the **same directory**:

   * `best.pt` (YOLOv8 model trained to detect: ball, goalkeeper, player, referee)
   * `15sec_input_720p.mp4` (input football video clip)

2. Run the complete Python script (`player_tracker.py`) on **Google Colab** or locally with the following environment setup.

3. The code will generate `output.mp4` as the final annotated video with player tracking and persistent IDs.

---

### ✅ Dependencies / Environment Requirements

* Python >= 3.8
* Packages:

  * `ultralytics`
  * `opencv-python-headless`
  * `numpy`

To install dependencies (Colab or local):

```bash
pip install ultralytics opencv-python-headless
```

---

