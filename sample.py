from vis_nav_game import Player, Action, Phase

import pygame
import cv2
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import faiss
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
DATA_DIR = "data/exploration_data"

# Graph construction
TEMPORAL_WEIGHT = 1.0       # edge weight for consecutive frames
VISUAL_WEIGHT_BASE = 2.0    # base weight for visual shortcut edges
VISUAL_WEIGHT_SCALE = 3.0   # weight += scale * vlad_distance
MIN_SHORTCUT_GAP = 50       # minimum trajectory index gap for shortcuts

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VLAD Feature Extraction
# ---------------------------------------------------------------------------
import torch
import timm
import numpy as np
import cv2
from tqdm import tqdm

class DinoV2Extractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        import timm

        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224   # 🔥 FORCE 224 INPUT
        ).to(self.device)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

    @property
    def dim(self):
        return 384  # ViT-S/14 output size

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # must match model
        img = img / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32)

    def extract(self, img):
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x)
        feat = feat.cpu().numpy().flatten()
        feat /= np.linalg.norm(feat) + 1e-8
        return feat

    def extract_batch(self, file_list):
        feats = []
        for fname in tqdm(file_list, desc="DINOv2"):
            img = cv2.imread(fname)
            if img is None:
                feats.append(np.zeros(self.dim))
                continue
            feats.append(self.extract(img))
        return np.array(feats, dtype="float32")
# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):

    def __init__(self, n_clusters: int = 128, subsample_rate: int = 5,
                 top_k_shortcuts: int = 30):
        self.recent_nodes = []
        self.smooth_window = 5
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super().__init__()

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        # Load trajectory data — supports both formats:
        #   New: data/traj_0/, data/traj_1/, ... each with data_info.json + images
        #   Legacy: data/images/ + data/data_info.json
        self.motion_frames = []   # list of {step, image, action, traj_id, image_path}
        self.file_list = []       # image paths relative to cwd
        self.traj_boundaries = [] # (start_idx, end_idx) per trajectory in motion_frames

        traj_dirs = sorted([
            d for d in os.listdir(DATA_DIR)
            if d.startswith('traj_') and os.path.isdir(os.path.join(DATA_DIR, d))
        ])

        if traj_dirs:
            # New multi-trajectory format
            all_motion = []
            for traj_dir_name in traj_dirs:
                traj_path = os.path.join(DATA_DIR, traj_dir_name)
                info_path = os.path.join(traj_path, 'data_info.json')
                if not os.path.exists(info_path):
                    continue
                with open(info_path) as f:
                    raw = json.load(f)
                traj_id = traj_dir_name
                pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
                traj_motion = [
                    {'step': d['step'], 'image': d['image'], 'action': d['action'][0],
                     'traj_id': traj_id, 'image_path': os.path.join(traj_path, d['image'])}
                    for d in raw
                    if len(d['action']) == 1 and d['action'][0] in pure
                ]
                start_idx = len(all_motion)
                all_motion.extend(traj_motion)
                end_idx = len(all_motion)
                self.traj_boundaries.append((start_idx, end_idx))
                print(f"  {traj_dir_name}: {len(traj_motion)} motion frames")

            self.motion_frames = all_motion[::subsample_rate]
            # Recompute boundaries after subsampling
            self.traj_boundaries = []
            prev_traj = None
            for idx, m in enumerate(self.motion_frames):
                if m['traj_id'] != prev_traj:
                    if prev_traj is not None:
                        self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], idx)
                    self.traj_boundaries.append((idx, len(self.motion_frames)))
                    prev_traj = m['traj_id']
            if self.traj_boundaries:
                self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], len(self.motion_frames))

            self.file_list = [m['image_path'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, "
                  f"{len(self.motion_frames)} after {subsample_rate}x subsample, "
                  f"{len(self.traj_boundaries)} trajectories")
        else:
            # Legacy single-directory format
            legacy_info = os.path.join(DATA_DIR, 'data_info.json')
            legacy_img_dir = os.path.join(DATA_DIR, 'images')
            if os.path.exists(legacy_info):
                with open(legacy_info) as f:
                    raw = json.load(f)
                pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
                all_motion = [
                    {'step': d['step'], 'image': d['image'], 'action': d['action'][0],
                     'traj_id': 'traj_0',
                     'image_path': os.path.join(legacy_img_dir, d['image'])}
                    for d in raw
                    if len(d['action']) == 1 and d['action'][0] in pure
                ]
                self.motion_frames = all_motion[::subsample_rate]
                self.file_list = [m['image_path'] for m in self.motion_frames]
                self.traj_boundaries = [(0, len(self.motion_frames))]
                print(f"Frames (legacy): {len(all_motion)} total, "
                      f"{len(self.motion_frames)} after {subsample_rate}x subsample")

        self.extractor = DinoV2Extractor()
        self.database = None
        self.G = None
        self.goal_node = None

    # --- Game engine hooks ---
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state and self._state[1] == Phase.NAVIGATION:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.display_next_best_view()

        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    def pre_navigation(self):
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()

    # --- VLAD database ---
    def _build_database(self):
        if self.database is not None:
            print("Database already computed, skipping.")
            return

        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    # --- Navigation graph ---

    def _build_graph(self):
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        # --- Temporal edges ---
        for start, end in self.traj_boundaries:
            for i in range(start, end - 1):
                self.G.add_edge(i, i + 1,
                                weight=TEMPORAL_WEIGHT,
                                edge_type="temporal")

        # --- FAISS for visual shortcuts ---
        print("Building FAISS index...")
        xb = self.database.astype("float32")
        faiss.normalize_L2(xb)

        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)

        K = 50  # neighbors per node
        D, I = index.search(xb, K)

        edges_added = 0
        dists = []

        print("Adding visual shortcut edges...")
        for i in range(n):
            for j_idx in range(1, K):  # skip self
                j = I[i, j_idx]

                if abs(i - j) < MIN_SHORTCUT_GAP:
                    continue

                s = float(D[i, j_idx])
                d = float(np.sqrt(max(0, 2 - 2 * s)))

                self.G.add_edge(
                    i, j,
                    weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d,
                    edge_type="visual"
                )

                dists.append(d)
                edges_added += 1

                if j_idx >= self.top_k_shortcuts:
                    break
            if j_idx >= self.top_k_shortcuts:
                break

        kd = np.array(dists)
        print(f"{edges_added} visual edges, dist: [{kd.min():.3f}, {kd.max():.3f}]")
        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # --- Goal ---

    def _setup_goal(self):
        """Set goal node from front-view target image."""
        if self.goal_node is not None:
            print("Goal already set, skipping.")
            return
        targets = self.get_target_images()
        if not targets:
            return
        sims = self.database @ self.extractor.extract(targets[0])
        self.goal_node = int(np.argmax(sims))
        d = float(np.sqrt(max(0, 2 - 2 * sims[self.goal_node])))
        print(f"Goal: node {self.goal_node} (d={d:.4f})")

    # --- Helpers ---
    def _load_img(self, idx: int) -> np.ndarray | None:
        """Load image by database index."""
        if 0 <= idx < len(self.file_list):
            return cv2.imread(self.file_list[idx])
        return None

    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        sims = (self.database @ feat) ** 2
        #sims = self.database @ feat

        # Top-k candidates (NOT just argmax)
        top_k = 5
        candidates = np.argpartition(-sims, top_k)[:top_k]

        # Pick best among candidates based on temporal consistency
        if len(self.recent_nodes) > 0:
            prev = self.recent_nodes[-1]

            # bias toward nearby nodes
            best = min(candidates, key=lambda c: abs(c - prev))
        else:
            best = candidates[np.argmax(sims[candidates])]

        # update buffer
        self.recent_nodes.append(best)
        if len(self.recent_nodes) > self.smooth_window:
            self.recent_nodes.pop(0)
        
        best_sim = np.max(sims)

        # reject bad matches
        if best_sim < 0.3:   # tune: 0.25–0.4
            if len(self.recent_nodes) > 0:
                return self.recent_nodes[-1]

        # majority vote smoothing
        return max(set(self.recent_nodes), key=self.recent_nodes.count)
    def _get_path(self, start: int) -> list[int]:
        """Shortest path from start to goal_node."""
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        """Get the action label for traversing edge a->b."""
        REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
                    'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]['action']
        elif b == a - 1 and b < len(self.motion_frames):
            return REVERSE.get(self.motion_frames[b]['action'], '?')
        return '?'

    # --- Display ---
    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                           ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        
        """
        Navigation panel:
            Info bar: current node | goal | hops | next action
            Row 1:    [Live FPV] [Best match] [Target (front)]
            Row 2:    Path preview (next 5 nodes)
        """
        ACT = {'FORWARD': 'FWD', 'BACKWARD': 'BACK', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT'}
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        TW, TH = 260, 195          # main thumbnails
        PW, PH = TW * 3 // 5, TH * 3 // 5   # path preview thumbnails
        N_PREVIEW = 5

        # Localize & plan
        cur = self._get_current_node()
        cur_sim = float(self.database[cur] @ self.extractor.extract(self.fpv))
        cur_d = float(np.sqrt(max(0, 2 - 2 * cur_sim)))
        path = self._get_path(cur)
        hops = len(path) - 1

        # --- Next waypoint image ---
        next_node = None
        if len(path) > 1:
            next_node = path[1]
        else:
            next_node = cur

        next_img = self._load_img(next_node)
        if next_img is None:
            next_img = np.zeros_like(self.fpv)

        next_img = cv2.resize(next_img, (TW, TH))

        cv2.rectangle(next_img, (0, 0), (TW-1, TH-1), (255, 200, 0), 2)
        cv2.putText(next_img, f"NEXT NODE {next_node}", (6, 22),
                    FONT, 0.5, (255, 200, 0), 1, AA)

        # Analyze edges
        edge_info = []
        for a, b in zip(path[:-1], path[1:]):
            et = self.G[a][b].get("edge_type", "temporal")
            if et == "temporal":
                act = ACT.get(self._edge_action(a, b), '?')
                edge_info.append(("seq", act, b == a + 1))
            else:
                edge_info.append(("vis", None, None))
        t_steps = sum(1 for e in edge_info if e[0] == "seq")
        v_jumps = len(edge_info) - t_steps

        if edge_info:
            etype, act, _ = edge_info[0]
            hint = act if etype == "seq" else "VISUAL JUMP"
        else:
            hint = "AT GOAL"
        near = hops <= 5

        # --- Info bar ---
        panel_w = TW * 3
        bar = np.zeros((40, panel_w, 3), dtype=np.uint8)
        bar[:] = (0, 0, 160) if near else (50, 35, 15)
        txt = (f"Node {cur} (d={cur_d:.3f})"
               f"  |  Goal {self.goal_node}"
               f"  |  {hops} hops ({t_steps}s+{v_jumps}v)"
               f"  |  >> {hint}")
        cv2.putText(bar, txt, (8, 27), FONT, 0.48, (255, 255, 255), 1, AA)
        if near:
            cv2.putText(bar, "NEAR TARGET — SPACE",
                        (panel_w - 220, 27), FONT, 0.48, (0, 255, 255), 1, AA)

        # --- Row 1: [FPV] [Match] [Target] ---
        def thumb(img, label, color, extra=None):
            t = cv2.resize(img, (TW, TH))
            cv2.rectangle(t, (0, 0), (TW-1, TH-1), color, 2)
            cv2.putText(t, label, (6, 22), FONT, 0.55, color, 1, AA)
            if extra:
                cv2.putText(t, extra, (6, 44), FONT, 0.45, (200, 200, 200), 1, AA)
            return t

        fpv_t = thumb(self.fpv, "Live FPV", (255, 255, 255))
        match_img = self._load_img(cur)
        if match_img is None:
            match_img = np.zeros((TH, TW, 3), dtype=np.uint8)
        match_t = thumb(match_img, f"Match: node {cur}", (0, 255, 0), f"d={cur_d:.3f}")
        targets = self.get_target_images()
        tgt = targets[0] if targets else np.zeros((TH, TW, 3), dtype=np.uint8)
        tgt_t = thumb(tgt, "Target (front)", (0, 140, 255))
        #row1 = cv2.hconcat([fpv_t, match_t, tgt_t])
        row1 = cv2.hconcat([fpv_t, match_t, next_img])

        # --- Row 2: path preview ---
        preview = path[1:1 + N_PREVIEW]
        cells = []
        for p in range(N_PREVIEW):
            if p < len(preview):
                img = self._load_img(preview[p])
                if img is None:
                    img = np.zeros((PH, PW, 3), dtype=np.uint8)
                img = cv2.resize(img, (PW, PH))
                etype, act, is_fwd = edge_info[p]
                if etype == "seq":
                    lbl = f"{'>' if is_fwd else '<'} {act}"
                    clr = (200, 200, 0)
                else:
                    lbl = "~ VISUAL"
                    clr = (200, 100, 255)
                cv2.rectangle(img, (0, 0), (PW-1, PH-1), clr, 1)
                cv2.putText(img, f"+{p+1} node {preview[p]}", (4, 16),
                            FONT, 0.38, (255, 255, 255), 1, AA)
                cv2.putText(img, lbl, (4, 34), FONT, 0.38, clr, 1, AA)
            else:
                img = np.zeros((PH, PW, 3), dtype=np.uint8)
            cells.append(img)
        row2 = cv2.hconcat(cells)

        # Pad row2 to match panel width
        if row2.shape[1] < panel_w:
            pad = np.zeros((PH, panel_w - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, pad])

        panel = cv2.vconcat([bar, row1, row2])
        cv2.imshow("Navigation", panel)
        cv2.waitKey(1)
        print(f"Node {cur} -> Goal {self.goal_node} | "
              f"{hops} hops ({t_steps}s+{v_jumps}v) | >> {hint}")


if __name__ == "__main__":
    import argparse
    import vis_nav_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=2,
                        help="Take every Nth motion frame (default: 2)")
    parser.add_argument("--n-clusters", type=int, default=128,
                        help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of global visual shortcut edges (default: 30)")
    args = parser.parse_args()

    vis_nav_game.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
    ))
