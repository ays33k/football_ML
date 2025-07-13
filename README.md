

## Project Title: **Player Tracking in Football Video Using YOLOv8 + IOU Matching**

---

### ✅ How to Set Up and Run the Code

1. Place the following files in the **same directory**:

   * `best.pt` (YOLOv8 model trained to detect: ball, goalkeeper, player, referee)
   * `15sec_input_720p.mp4` (input football video clip)
   * `player_tracker.py` (main script)

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script on **Google Colab** or locally:

```bash
python player_tracker.py
```

4. The code will generate `output.mp4` as the final annotated video with player tracking and persistent IDs.





---

