"""
Working VLAD baseline for vis_nav_player.

Pipeline:
RootSIFT -> KMeans codebook -> VLAD -> graph search -> target matching
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import pickle
import networkx as nx

from sklearn.cluster import KMeans
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DATA_DIR = os.path.join(BASE_DIR, "data", "exploration_data")
LEGACY_IMAGE_DIR = os.path.join(DATA_DIR, "images")
LEGACY_INFO_PATH = os.path.join(DATA_DIR, "data_info.json")

TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 50

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VLAD feature extractor
# ---------------------------------------------------------------------------
class VLADExtractor:
    """RootSIFT + VLAD with intra-normalization and power normalization."""

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        eps = 1e-12
        des = des / (np.sum(des, axis=1, keepdims=True) + eps)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters

        vlad = np.zeros((k, des.shape[1]), dtype=np.float32)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm

        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    def load_sift_cache(self, file_list: list[str], subsample_rate: int):
        cache_file = os.path.join(CACHE_DIR, f"sift_ss{subsample_rate}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in file_list):
                return
            print("  Cache incomplete, re-extracting...")

        print(f"Extracting SIFT for {len(file_list)} images...")
        self._sift_cache = {}

        for fname in tqdm(file_list, desc="SIFT"):
            img = cv2.imread(fname)
            if img is None:
                continue
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None and len(des) > 0:
                self._sift_cache[fname] = self._root_sift(des)

        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)

        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self, file_list: list[str]):
        cache_file = os.path.join(CACHE_DIR, f"codebook_k{self.n_clusters}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        valid_desc = [self._sift_cache[f] for f in file_list if f in self._sift_cache]
        if not valid_desc:
            raise ValueError("No valid SIFT descriptors found to build vocabulary.")

        all_des = np.vstack(valid_desc)
        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")

        self.codebook = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=3,
            max_iter=300,
            tol=1e-4,
            verbose=1,
            random_state=42,
        ).fit(all_des)

        print(f"  {self.codebook.n_iter_} iters, inertia={self.codebook.inertia_:.0f}")

        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list[str]) -> np.ndarray:
        vectors = []
        for fname in tqdm(file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
        return np.array(vectors)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):
    def __init__(self, n_clusters: int = 128, subsample_rate: int = 5,
                 top_k_shortcuts: int = 30):
        super().__init__()

        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.database = None
        self.G = None
        self.goal_node = None

        self.motion_frames = []      # list of dicts
        self.file_list = []          # full image paths
        self.traj_boundaries = []    # (start, end)

        self._load_trajectory_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_action(action):
        valid = {"FORWARD", "LEFT", "RIGHT", "BACKWARD", "IDLE"}

        if action is None:
            return "IDLE"

        if isinstance(action, str):
            s = action.strip().upper()
            if "." in s:
                s = s.split(".")[-1]
            return s if s in valid else "IDLE"

        if isinstance(action, (list, tuple, set)):
            for item in action:
                if isinstance(item, str):
                    s = item.strip().upper()
                    if "." in s:
                        s = s.split(".")[-1]
                    if s in valid:
                        return s
            return "IDLE"

        s = str(action).strip().upper()
        if "." in s:
            s = s.split(".")[-1]
        return s if s in valid else "IDLE"

    def _load_trajectory_data(self):
        traj_dirs = sorted([
            d for d in os.listdir(DATA_DIR)
            if d.startswith("traj_") and os.path.isdir(os.path.join(DATA_DIR, d))
        ]) if os.path.exists(DATA_DIR) else []

        if traj_dirs:
            print("Detected multi-trajectory format")
            all_frames = []

            for traj_name in traj_dirs:
                traj_path = os.path.join(DATA_DIR, traj_name)
                info_path = os.path.join(traj_path, "data_info.json")

                if not os.path.exists(info_path):
                    continue

                with open(info_path, "r") as f:
                    raw = json.load(f)

                start_before = len(all_frames)

                for i, d in enumerate(raw):
                    image_name = d.get("image")
                    if image_name is None:
                        continue

                    image_path = os.path.join(traj_path, image_name)
                    all_frames.append({
                        "step": d.get("step", i),
                        "image": image_name,
                        "image_path": image_path,
                        "action": self._normalize_action(d.get("action")),
                        "traj_id": traj_name,
                    })

                end_after = len(all_frames)
                print(f"  {traj_name}: {end_after - start_before} frames")

            self.motion_frames = all_frames[::self.subsample_rate]
            self.file_list = [m["image_path"] for m in self.motion_frames]

            self.traj_boundaries = []
            prev_traj = None
            start_idx = 0
            for idx, frame in enumerate(self.motion_frames):
                if prev_traj is None:
                    prev_traj = frame["traj_id"]
                    start_idx = idx
                elif frame["traj_id"] != prev_traj:
                    self.traj_boundaries.append((start_idx, idx))
                    prev_traj = frame["traj_id"]
                    start_idx = idx
            if self.motion_frames:
                self.traj_boundaries.append((start_idx, len(self.motion_frames)))

            print(f"Frames: {len(all_frames)} total, {len(self.motion_frames)} after {self.subsample_rate}x subsample")

        else:
            print("Detected legacy single-trajectory format")

            if not os.path.exists(LEGACY_INFO_PATH):
                raise ValueError(f"Could not find trajectory metadata: {LEGACY_INFO_PATH}")

            with open(LEGACY_INFO_PATH, "r") as f:
                raw = json.load(f)

            all_frames = []
            for i, d in enumerate(raw):
                image_name = d.get("image")
                if image_name is None:
                    continue

                all_frames.append({
                    "step": d.get("step", i),
                    "image": image_name,
                    "image_path": os.path.join(LEGACY_IMAGE_DIR, image_name),
                    "action": self._normalize_action(d.get("action")),
                    "traj_id": "traj_0",
                })

            self.motion_frames = all_frames[::self.subsample_rate]
            self.file_list = [m["image_path"] for m in self.motion_frames]
            self.traj_boundaries = [(0, len(self.motion_frames))]

            print(f"Frames (legacy): {len(all_frames)} total, {len(self.motion_frames)} after {self.subsample_rate}x subsample")

        if not self.file_list:
            raise ValueError("No usable exploration frames were loaded.")

    # ------------------------------------------------------------------
    # Game engine hooks
    # ------------------------------------------------------------------
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
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")
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

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    def _build_database(self):
        if self.database is not None:
            print("Database already computed, skipping.")
            return

        self.extractor.load_sift_cache(self.file_list, self.subsample_rate)
        self.extractor.build_vocabulary(self.file_list)
        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _build_graph(self):
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        if self.database is None:
            raise ValueError("Build database before building graph.")

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        # Attach metadata
        for i, frame in enumerate(self.motion_frames):
            self.G.nodes[i]["image"] = frame["image"]
            self.G.nodes[i]["step"] = frame["step"]
            self.G.nodes[i]["action"] = frame["action"]
            self.G.nodes[i]["traj_id"] = frame["traj_id"]

        # ------------------------------------------------------------------
        # Temporal edges
        # ------------------------------------------------------------------
        for start, end in self.traj_boundaries:
            for i in range(start, end - 1):
                self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        # ------------------------------------------------------------------
        # k-NN + mutual + threshold visual edges
        # ------------------------------------------------------------------
        print(f"Building k-NN graph (k={self.top_k_shortcuts}, min_gap={MIN_SHORTCUT_GAP})...")

        k = self.top_k_shortcuts
        SIM_THRESHOLD = 0.6  # adjust if needed

        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -np.inf)

        # Apply temporal exclusion window
        for i in range(n):
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim[i, lo:hi] = -np.inf

        # Step 1: compute k-NN sets
        neighbors = []
        for i in range(n):
            nn_idx = np.argpartition(-sim[i], k)[:k]
            neighbors.append(set(nn_idx))

        # Step 2: add ONLY mutual edges above threshold
        edge_count = 0
        dists = []

        for i in range(n):
            for j in neighbors[i]:
                if i >= j:
                    continue

                # mutual condition
                if i not in neighbors[j]:
                    continue

                s = float(sim[i, j])
                if not np.isfinite(s) or s < SIM_THRESHOLD:
                    continue

                d = float(np.sqrt(max(0, 2 - 2 * s)))

                self.G.add_edge(
                    i,
                    j,
                    weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d,
                    edge_type="visual",
                )

                dists.append(d)
                edge_count += 1

        if dists:
            kd = np.array(dists)
            print(f"  {edge_count} visual edges, dist: [{kd.min():.3f}, {kd.max():.3f}]")

        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Goal / helpers
    # ------------------------------------------------------------------
    def _setup_goal(self):
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

    def _load_img(self, idx: int):
        if 0 <= idx < len(self.file_list):
            return cv2.imread(self.file_list[idx])
        return None

    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        return int(np.argmax(self.database @ feat))

    def _get_path(self, start: int) -> list[int]:
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        reverse = {
            "FORWARD": "BACKWARD",
            "BACKWARD": "FORWARD",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
            "IDLE": "IDLE",
        }

        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]["action"]
        elif b == a - 1 and b < len(self.motion_frames):
            return reverse.get(self.motion_frames[b]["action"], "?")
        return "?"

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
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
        for label, pos in [
            ("Front", (10, 25)),
            ("Right", (w // 2 + 10, 25)),
            ("Back", (10, h // 2 + 25)),
            ("Left", (w // 2 + 10, h // 2 + 25)),
        ]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Target Images", img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        ACT = {"FORWARD": "FWD", "BACKWARD": "BACK", "LEFT": "LEFT", "RIGHT": "RIGHT"}
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        TW, TH = 260, 195
        PW, PH = TW * 3 // 5, TH * 3 // 5
        N_PREVIEW = 5

        cur = self._get_current_node()
        cur_sim = float(self.database[cur] @ self.extractor.extract(self.fpv))
        cur_d = float(np.sqrt(max(0, 2 - 2 * cur_sim)))
        path = self._get_path(cur)
        hops = len(path) - 1

        edge_info = []
        for a, b in zip(path[:-1], path[1:]):
            et = self.G[a][b].get("edge_type", "temporal")
            if et == "temporal":
                act = ACT.get(self._edge_action(a, b), "?")
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

        panel_w = TW * 3
        bar = np.zeros((40, panel_w, 3), dtype=np.uint8)
        bar[:] = (0, 0, 160) if near else (50, 35, 15)
        txt = (
            f"Node {cur} (d={cur_d:.3f})"
            f"  |  Goal {self.goal_node}"
            f"  |  {hops} hops ({t_steps}s+{v_jumps}v)"
            f"  |  >> {hint}"
        )
        cv2.putText(bar, txt, (8, 27), FONT, 0.48, (255, 255, 255), 1, AA)

        if near:
            cv2.putText(bar, "NEAR TARGET — SPACE", (panel_w - 220, 27),
                        FONT, 0.48, (0, 255, 255), 1, AA)

        def thumb(img, label, color, extra=None):
            t = cv2.resize(img, (TW, TH))
            cv2.rectangle(t, (0, 0), (TW - 1, TH - 1), color, 2)
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

        row1 = cv2.hconcat([fpv_t, match_t, tgt_t])

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

                cv2.rectangle(img, (0, 0), (PW - 1, PH - 1), clr, 1)
                cv2.putText(img, f"+{p+1} node {preview[p]}", (4, 16),
                            FONT, 0.38, (255, 255, 255), 1, AA)
                cv2.putText(img, lbl, (4, 34), FONT, 0.38, clr, 1, AA)
            else:
                img = np.zeros((PH, PW, 3), dtype=np.uint8)

            cells.append(img)

        row2 = cv2.hconcat(cells)

        if row2.shape[1] < panel_w:
            pad = np.zeros((PH, panel_w - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, pad])

        panel = cv2.vconcat([bar, row1, row2])
        cv2.imshow("Navigation", panel)
        cv2.waitKey(1)

        print(f"Node {cur} -> Goal {self.goal_node} | {hops} hops ({t_steps}s+{v_jumps}v) | >> {hint}")


if __name__ == "__main__":
    import argparse
    import vis_nav_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=5,
                        help="Take every Nth motion frame (default: 5)")
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
