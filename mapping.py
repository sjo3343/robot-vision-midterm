import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import glob

# Load all maze FPV images in order
image_dir = 'traj_0'
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
all_frames = [cv2.imread(p) for p in image_paths]

height, width = all_frames[0].shape[:2]
print(f"Loaded {len(all_frames)} frames from {image_dir}/")
print(f"Resolution: {width} x {height}")

# Show the first frame
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2RGB))
plt.title("Frame 0 — Robot's first-person view in the maze")
plt.axis('off')
plt.show()

# Detect Shi-Tomasi corners
feature_params = dict(
    maxCorners=100,    # Maximum number of corners to detect
    qualityLevel=0.1,  # Minimum quality (relative to the best corner)
    minDistance=7,      # Minimum distance between detected corners
    blockSize=7         # Size of the neighborhood for corner detection
)

lk_params = dict(
    winSize=(15, 15),   # Size of the search window around each point
    maxLevel=2,         # Number of pyramid levels (more on this later!)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Full KLT loop WITH re-detection every M frames
REDETECT_INTERVAL = 10

old_gray = cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2GRAY)
#sift=cv2.SIFT_create()
#p0,_=sift.detectAndCompute(old_gray, None)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#points = np.float32(cv2.KeyPoint_convert(p0))

#print(points)
frames_redetect = []
feature_counts_redetect = []

color_map = np.random.randint(0, 255, (5000, 3))
next_color_idx = len(p0)
track_colors = [color_map[i].tolist() for i in range(len(p0))]
track_histories = [[tuple(p.ravel())] for p in p0]
print(track_histories)
for frame_idx, frame in enumerate(all_frames[1:], 1):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        break

    good_mask = st.ravel() == 1
    good_new = p1[good_mask]

    track_colors = [c for c, ok in zip(track_colors, good_mask) if ok]
    track_histories = [h for h, ok in zip(track_histories, good_mask) if ok]
    for i, pt in enumerate(good_new):
        if i >= len(track_histories):
            break
        track_histories[i].append(tuple(pt.ravel()))

    # Re-detect every M frames
    if frame_idx % REDETECT_INTERVAL == 0 and len(good_new) > 0:
        new_corners = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if new_corners is not None:
            existing = good_new.reshape(-1, 2)
            for pt in new_corners:
                x, y = pt.ravel()
                dists = np.sqrt((existing[:, 0] - x)**2 + (existing[:, 1] - y)**2)
                if dists.min() > 10:
                    good_new = np.vstack([good_new.reshape(-1, 1, 2), pt.reshape(1, 1, 2)])
                    track_colors.append(color_map[next_color_idx % 5000].tolist())
                    track_histories.append([(x, y)])
                    next_color_idx += 1

    feature_counts_redetect.append(len(good_new))

    # Draw only active tracks
    display = frame.copy()
    for i, hist in enumerate(track_histories):
        c = track_colors[i]
        for j in range(1, len(hist)):
            cv2.line(display, (int(hist[j-1][0]), int(hist[j-1][1])),
                     (int(hist[j][0]), int(hist[j][1])), c, 2)
        x, y = hist[-1]
        cv2.circle(display, (int(x), int(y)), 4, c, -1)

    frames_redetect.append(display)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

print(f"Tracked {len(frames_redetect)} frames with re-detection")

# Classify robot action from optical flow in each frame
prev_gray = cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
h, w = all_frames[0].shape[:2]
cx, cy = w / 2, h / 2

actions = []
action_data = []  # (avg_dx, divergence) for plotting

for frame in all_frames[1:]:
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    good_mask = st.ravel() == 1
    old = prev_pts[good_mask].reshape(-1, 2)
    new = curr_pts[good_mask].reshape(-1, 2)

    if len(old) > 0:
        dx = (new[:, 0] - old[:, 0]).mean()
        old_dist = np.sqrt((old[:, 0] - cx)**2 + (old[:, 1] - cy)**2)
        new_dist = np.sqrt((new[:, 0] - cx)**2 + (new[:, 1] - cy)**2)
        divergence = (new_dist - old_dist).mean()
    else:
        dx, divergence = 0, 0

    action_data.append((dx, divergence))

    if abs(dx) < 1.5 and abs(divergence) < 2:
        action = "IDLE"
    elif divergence > 4 and abs(dx) < 3:
        action = "FORWARD"
    elif dx > 3:
        action = "TURN LEFT"
    elif dx < -3:
        action = "TURN RIGHT"
    elif divergence > 3:
        action = "FORWARD"
    else:
        action = "FORWARD" if divergence > 0 else "IDLE"

    actions.append(action)

    prev_gray = curr_gray
    new_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
    prev_pts = new_pts if new_pts is not None else curr_pts[good_mask].reshape(-1, 1, 2)


from collections import Counter
counts = Counter(actions)
for action, count in counts.most_common():
    print(f"  {action}: {count} frames ({count/len(actions)*100:.0f}%)")

# Plot action timeline + the underlying flow signals
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

dx_vals = [d[0] for d in action_data]
div_vals = [d[1] for d in action_data]
frames_range = range(1, len(actions) + 1)

axes[0].plot(frames_range, dx_vals, 'b-', alpha=0.7, linewidth=0.8)
axes[0].axhline(y=3, color='r', linestyle='--', alpha=0.3, label='Turn threshold')
axes[0].axhline(y=-3, color='r', linestyle='--', alpha=0.3)
axes[0].set_ylabel('Avg dx (pixels)')
axes[0].set_title('Horizontal Flow → Left/Right Turn Detection')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(frames_range, div_vals, 'g-', alpha=0.7, linewidth=0.8)
axes[1].axhline(y=4, color='r', linestyle='--', alpha=0.3, label='Forward threshold')
axes[1].set_ylabel('Divergence')
axes[1].set_title('Flow Divergence → Forward Motion Detection')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Action timeline as colored bar
action_to_num = {"IDLE": 0, "FORWARD": 1, "TURN LEFT": 2, "TURN RIGHT": 3}
action_nums = [action_to_num[a] for a in actions]
colors_list = ['gray', 'green', 'dodgerblue', 'orange']
cmap = plt.matplotlib.colors.ListedColormap(colors_list)
axes[2].imshow([action_nums], aspect='auto', cmap=cmap, vmin=0, vmax=3,
               extent=[1, len(actions), 0, 1])
axes[2].set_yticks([])
axes[2].set_xlabel('Frame number')
axes[2].set_title('Detected Action Timeline')
# Legend
for name, c in zip(["Idle", "Forward", "Turn Left", "Turn Right"], colors_list):
    axes[2].plot([], [], 's', color=c, label=name, markersize=10)
axes[2].legend(loc='upper right', ncol=4)

plt.tight_layout()
plt.show()

import turtle
tutle=turtle.Turtle()
window=turtle.Screen()
window.screensize(1000,800)
tutle.penup()
#tutle.goto(-450,350)
tutle.pendown()
counter=0
for actions in action_nums:
    #goal frame
    tutle.pencolor('red') if abs(counter-27057)<15 else tutle.pencolor('black')
    match actions:
        case 1:
            tutle.forward(2)
        case 2:
            tutle.left(2.1)
        case 3:
            tutle.right(2.1)
    counter+=1
