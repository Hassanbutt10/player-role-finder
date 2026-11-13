"""
Player Role Finder - Cycle 11, Day 6 project
What does a player do on the ball, and who else at Copa America 2024 does the
same? Every outfield regular is described by 12 on-ball features, grouped
into roles with k-means, and checked against the position the data lists for
him. A nearest-neighbour search then answers "who plays like Messi?", and a
re-sampling test shows how far those answers can be trusted.

Method:
  1. Periods 1 to 4. An action is a pass, carry, dribble, shot, ball receipt,
     pressure, duel, interception, block, clearance, ball recovery, foul,
     dispossession, miscontrol or dribbled-past event.
  2. Players with at least 300 actions in total are kept. Goalkeepers are left
     out (their role is obvious from the data). There are no minutes in the
     data, so every feature is a share of the player's own actions or a
     position, never a per-90 rate. This describes a role, not quality.
  3. Twelve features: share of actions that are passes, carries, dribbles,
     shots, pressures, defensive actions (duel, interception, block, clearance,
     ball recovery) and ball receipts; average field position (yards from own
     goal line); wideness (average distance from the centre line); % of passes
     that are long (30+ yards), forward (10+ yards gained) and crosses.
  4. Features are z-scored. PCA (NumPy SVD) gives the 2D map, k-means (NumPy,
     k-means++ starts) groups the players, and k from 3 to 8 is chosen by
     silhouette score.
  5. Check: each player's listed position is mapped to 7 groups (centre-back,
     full-back, defensive midfield, central midfield, attacking midfield,
     winger, striker). The adjusted Rand index (ARI) compares roles with those
     groups.
  6. Lookup: the closest players are the smallest distances in the 12-feature
     z-score space.
  7. Reliability: each player's matches are re-sampled 500 times, features
     rebuilt, players re-grouped and neighbours recomputed. For each lookup
     the share of re-samples in which a neighbour stays in the top 5 is shown.

Limitations: no minutes, so a player with few actions has noisy shares (300
actions is about 3 full matches). Features are partly shaped by the team's
style and opponents, so team-mates look alike more often. Listed positions
are the most common position in the data, not a ground truth. Roles are not
named: a role is a summary of the 12 features. Distances on the 2D map are
approximate. A close neighbour means a similar action mix, not a similar
level of quality.

Outputs: player_role_map.png, player_role_lookup.png,
player_role_replay.mp4 (lookup replay plus the two charts as end cards),
player_roles.csv, player_neighbours.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import itertools
import os
import subprocess
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

MIN_ACTIONS = 300
K_RANGE = range(3, 9)
N_BOOT = 500
SEED = 9
TOP_N = 5
LONG_PASS = 30
FORWARD_GAIN = 10
END_CARDS = ["player_role_map.png", "player_role_lookup.png"]
END_CARD_SECONDS = 4

# players to look up: (name fragment, team)
QUERIES = [("messi", "Argentina"), ("enzo", "Argentina"), ("de paul", "Argentina"),
           ("romero", "Argentina"), ("di maria", "Argentina"), ("lautaro", "Argentina"),
           ("mac allister", "Argentina"), ("lisandro", "Argentina")]
VIDEO_QUERIES = 6

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"
PURPLE = "#b39ddb"
TEAL = "#4dd0e1"
PINK = "#f48fb1"
YELLOW = "#fff176"
ROLE_COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, PINK, YELLOW]

TEAM_CODES = {
    "Argentina": "ARG", "Brazil": "BRA", "Colombia": "COL", "Uruguay": "URU", "Canada": "CAN",
    "Venezuela": "VEN", "Ecuador": "ECU", "Panama": "PAN", "Mexico": "MEX", "United States": "USA",
    "Chile": "CHI", "Peru": "PER", "Paraguay": "PAR", "Bolivia": "BOL", "Jamaica": "JAM",
    "Costa Rica": "CRC"}
FEATS = ["pass_share", "carry_share", "drib_share", "shot_share", "press_share", "def_share", "recv_share",
         "avg_x", "wide", "long_pct", "fwd_pct", "cross_pct"]
FEAT_NAME = {"pass_share": "passing", "carry_share": "carrying", "drib_share": "dribbling",
             "shot_share": "shooting", "press_share": "pressing", "def_share": "defending",
             "recv_share": "receiving", "avg_x": "high position", "wide": "wide position",
             "long_pct": "long passes", "fwd_pct": "forward passes", "cross_pct": "crosses"}
POS_GROUPS = ["Centre-back", "Full-back", "Defensive mid", "Central mid", "Attacking mid", "Winger", "Striker"]

OVERRIDES = {
    "Lionel Andrés Messi Cuccittini": "Messi", "Ángel Fabián Di María Hernández": "Di María",
    "Nahuel Molina Lucero": "Molina", "Damián Emiliano Martínez": "E. Martínez",
    "Lautaro Javier Martínez": "Lautaro Martínez", "Lisandro Martínez": "Lisandro Martínez",
    "Raphael Dias Belloli": "Raphinha", "Luis Fernando Díaz Marulanda": "Luis Díaz",
    "Marcos Aoás Corrêa": "Marquinhos", "Marcos Aoas Correa": "Marquinhos",
    "Éderson José dos Santos Lourenço da Silva": "Ederson",
    "Enzo Jeremías Fernández": "Enzo Fernández", "Enzo Fernandez": "Enzo Fernández",
    "Rodrigo Javier De Paul": "De Paul",
    "Alexis Mac Allister": "Mac Allister", "Cristian Gabriel Romero": "Romero",
    "Vinícius José Paixão de Oliveira Júnior": "Vinícius Júnior",
    "Rodrygo Silva de Goes": "Rodrygo",
    "Bruno Guimarães Rodriguez Moura": "Bruno Guimarães",
    "Igor Lichnovsky Osorio": "Lichnovsky",
    "Jonathan Osorio": "J. Osorio",
    "Yangel Clemente Herrera Ravelo": "Yangel Herrera",
    "José Fajardo Nelson": "Fajardo",
}
PARTICLES = {"de", "da", "di", "dos", "van", "von", "mac", "lo", "la", "del", "le"}


def short_name(full):
    if full in OVERRIDES:
        return OVERRIDES[full]
    parts = full.split()
    for i in range(1, len(parts) - 1):
        if parts[i].lower() in PARTICLES:
            j = i
            while j < len(parts) - 1 and parts[j].lower() in PARTICLES:
                j += 1
            surname = " ".join(parts[i:j + 1])
            if surname.lower() == "da silva":
                return parts[0] if i == 2 else f"{parts[0]} {parts[i - 1]}"
            return surname
    if len(parts) >= 4:
        return parts[-2]
    return parts[-1]


def display_names(full_names):
    uniq = list(dict.fromkeys(full_names))
    base = {p: short_name(p) for p in uniq}
    groups = {}
    for p, b in base.items():
        groups.setdefault(b, []).append(p)
    out = {}
    for p, b in base.items():
        first = p.split()[0]
        out[p] = f"{first} {b}" if len(groups[b]) > 1 and not b.startswith(first) else b
    return out


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))


def pos_group(pos):
    p = str(pos)
    if "Goalkeeper" in p:
        return "GK"
    if "Center Back" in p:
        return "Centre-back"
    if "Back" in p:
        return "Full-back"
    if "Defensive Midfield" in p:
        return "Defensive mid"
    if "Attacking Midfield" in p:
        return "Attacking mid"
    if "Midfield" in p:
        return "Central mid"
    if "Wing" in p:
        return "Winger"
    if "Forward" in p or "Striker" in p:
        return "Striker"
    return "Other"


matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2, 3, 4])].copy()
for col in ["pass_length", "pass_cross", "location", "pass_end_location", "position"]:
    if col not in events.columns:
        events[col] = np.nan


def parse(col, a, b):
    xy = events[col].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
    events[a] = pd.to_numeric(xy[0], errors="coerce")
    events[b] = pd.to_numeric(xy[1], errors="coerce")


parse("location", "lx", "ly")
parse("pass_end_location", "ex", "ey")

ACTION_TYPES = ["Pass", "Carry", "Dribble", "Shot", "Ball Receipt*", "Pressure", "Duel", "Interception", "Block",
                "Clearance", "Ball Recovery", "Foul Committed", "Foul Won", "Dispossessed", "Miscontrol",
                "Dribbled Past"]
DEF_TYPES = ["Duel", "Interception", "Block", "Clearance", "Ball Recovery"]
a = events[events["type"].isin(ACTION_TYPES) & events["player"].notna()].copy()
a["is_pass"] = a["type"] == "Pass"
a["is_carry"] = a["type"] == "Carry"
a["is_drib"] = a["type"] == "Dribble"
a["is_shot"] = a["type"] == "Shot"
a["is_press"] = a["type"] == "Pressure"
a["is_def"] = a["type"].isin(DEF_TYPES)
a["is_recv"] = a["type"] == "Ball Receipt*"
a["is_long"] = a["is_pass"] & (a["pass_length"] >= LONG_PASS)
a["is_fwd"] = a["is_pass"] & ((a["ex"] - a["lx"]) >= FORWARD_GAIN)
a["is_cross"] = a["is_pass"] & a["pass_cross"].astype(str).str.lower().eq("true")
a["has_loc"] = a["lx"].notna()
a["x_val"] = a["lx"].fillna(0.0) * a["has_loc"]
a["absy_val"] = (a["ly"] - 40).abs().fillna(0.0) * a["has_loc"]
a["n_act"] = 1
SUMCOLS = ["n_act", "is_pass", "is_carry", "is_drib", "is_shot", "is_press", "is_def", "is_recv",
           "is_long", "is_fwd", "is_cross", "has_loc", "x_val", "absy_val"]
pm = a.groupby(["player", "team", "match_id"])[SUMCOLS].sum().astype(float).reset_index()
COLS = SUMCOLS
ci = {c: i for i, c in enumerate(COLS)}

modal_pos = (events.dropna(subset=["player", "position"]).groupby("player")["position"]
             .agg(lambda s: s.mode().iloc[0]))
tot = pm.groupby("player")["n_act"].sum()
keep = [p for p in tot[tot >= MIN_ACTIONS].index if pos_group(modal_pos.get(p, "")) != "GK"]
team_of = pm.drop_duplicates("player").set_index("player")["team"]
players = sorted(keep, key=lambda p: (team_of[p], p))
nP = len(players)
if nP < 30:
    raise ValueError(f"Only {nP} players have {MIN_ACTIONS}+ actions, lower MIN_ACTIONS")
rows_of = {p: np.where(pm["player"].to_numpy() == p)[0] for p in players}
CNT = pm[COLS].to_numpy(dtype=float)
n_matches = np.array([len(rows_of[p]) for p in players])
print(f"\n{nP} outfield players with {MIN_ACTIONS}+ actions (goalkeepers left out), "
      f"{int(tot[players].sum()):,} actions, {n_matches.min()} to {n_matches.max()} matches each")


def safe(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def features_from(S):
    g = lambda c: S[:, ci[c]]
    F = np.column_stack([
        safe(g("is_pass"), g("n_act")) * 100, safe(g("is_carry"), g("n_act")) * 100,
        safe(g("is_drib"), g("n_act")) * 100, safe(g("is_shot"), g("n_act")) * 100,
        safe(g("is_press"), g("n_act")) * 100, safe(g("is_def"), g("n_act")) * 100,
        safe(g("is_recv"), g("n_act")) * 100,
        safe(g("x_val"), g("has_loc")), safe(g("absy_val"), g("has_loc")),
        safe(g("is_long"), g("is_pass")) * 100, safe(g("is_fwd"), g("is_pass")) * 100,
        safe(g("is_cross"), g("is_pass")) * 100])
    col_mean = np.nanmean(F, axis=0)
    return np.where(np.isnan(F), col_mean, F)


S0 = np.vstack([CNT[rows_of[p]].sum(axis=0) for p in players])
X0 = features_from(S0)


def zscore(X):
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)


def kmeans(Z, k, rng, n_init=30, max_iter=100):
    n = len(Z)
    best = None
    for _ in range(n_init):
        idx = [int(rng.integers(n))]
        for _ in range(1, k):
            d2 = np.min(((Z[:, None, :] - Z[idx][None, :, :]) ** 2).sum(2), axis=1)
            pr_ = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            idx.append(int(rng.choice(n, p=pr_)))
        C = Z[idx].copy()
        labels = None
        for _ in range(max_iter):
            d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2)
            new = d.argmin(1)
            if labels is not None and (new == labels).all():
                break
            labels = new
            C = np.array([Z[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        inertia = float(((Z - C[labels]) ** 2).sum())
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, labels.copy(), C.copy())
    return best


def silhouette(Z, labels):
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(2))
    s = np.zeros(len(Z))
    for i in range(len(Z)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a_ = D[i, same].mean()
        b_ = min(D[i, labels == c].mean() for c in set(labels.tolist()) if c != labels[i])
        s[i] = (b_ - a_) / max(a_, b_)
    return float(s.mean())


def ari(x, y):
    n = len(x)
    cx, cy = sorted(set(x.tolist())), sorted(set(y.tolist()))
    M = np.array([[np.sum((x == u) & (y == v)) for v in cy] for u in cx], dtype=float)
    c2 = lambda v: v * (v - 1) / 2
    sum_ij = c2(M).sum()
    sa, sb = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    exp = sa * sb / c2(n)
    den = 0.5 * (sa + sb) - exp
    return 1.0 if den == 0 else float((sum_ij - exp) / den)


rng = np.random.default_rng(SEED)
Z0 = zscore(X0)
U, Sv, Vt = np.linalg.svd(Z0, full_matrices=False)
V2 = Vt[:2].T
for j in range(2):
    if V2[np.abs(V2[:, j]).argmax(), j] < 0:
        V2[:, j] *= -1
scores = Z0 @ V2
expl = Sv ** 2 / np.sum(Sv ** 2)
print(f"PCA: PC1 {expl[0] * 100:.0f}% and PC2 {expl[1] * 100:.0f}% of the variance in the 12 features")

print("\nChoosing k by silhouette score:")
best_k, best_sil, fits = None, -9, {}
for k in K_RANGE:
    fit = kmeans(Z0, k, rng, n_init=60)
    sil = silhouette(Z0, fit[1])
    fits[k] = fit
    print(f"  k={k}: silhouette {sil:.3f}")
    if sil > best_sil:
        best_k, best_sil = k, sil
K = best_k
lab_raw = fits[K][1]
order_ids = sorted(range(K), key=lambda c: X0[lab_raw == c, FEATS.index("avg_x")].mean())
mp = np.zeros(K, dtype=int)
for new, old in enumerate(order_ids):
    mp[old] = new
labels0 = mp[lab_raw]
print(f"Using k={K} (silhouette {best_sil:.3f}). Roles are numbered from deepest to highest average position.")

pg = np.array([pos_group(modal_pos.get(p, "")) for p in players])
pg_codes = {g: i for i, g in enumerate(POS_GROUPS)}
pg_num = np.array([pg_codes.get(g, len(POS_GROUPS)) for g in pg])
ari_pos = ari(labels0, pg_num)
cont = np.array([[int(((labels0 == c) & (pg == g)).sum()) for g in POS_GROUPS] for c in range(K)])
purity = cont.max(axis=1).sum() / cont.sum()
print(f"\nRoles vs listed position groups: adjusted Rand index {ari_pos:.2f}, purity {purity * 100:.0f}% "
      f"(share of players in the most common position group of their role)")

names = display_names(players)
codes = [TEAM_CODES.get(team_of[p], team_of[p][:3].upper()) for p in players]
print("\nRoles (counts by listed position):")
role_desc = []
for c in range(K):
    mem = np.where(labels0 == c)[0]
    zc = Z0[mem].mean(0)
    hi = [FEAT_NAME[FEATS[j]] for j in np.argsort(-zc)[:2] if zc[j] > 0.4]
    lo = [FEAT_NAME[FEATS[j]] for j in np.argsort(zc)[:2] if zc[j] < -0.4]
    comp = sorted(((int(cont[c, j]), POS_GROUPS[j]) for j in range(len(POS_GROUPS)) if cont[c, j] > 0), reverse=True)
    comp_txt = ", ".join(f"{n} {g.lower()}" for n, g in comp[:3])
    role_desc.append((len(mem), comp_txt, hi, lo))
    print(f"  Role {c + 1} ({len(mem)} players): {comp_txt}")
    print(f"     higher: {', '.join(hi) or 'none'}; lower: {', '.join(lo) or 'none'}")

# ---------- queries and neighbours ----------
Dm = np.sqrt(((Z0[:, None] - Z0[None]) ** 2).sum(2))
q_idx = []
for token, team in QUERIES:
    cand = [i for i, p in enumerate(players) if norm(token) in norm(p) and (team is None or team_of[p] == team)]
    if not cand:
        print(f"Query '{token}' ({team}) not found among the {nP} players")
        continue
    q_idx.append(max(cand, key=lambda i: tot[players[i]]))
q_idx = list(dict.fromkeys(q_idx))
if not q_idx:
    q_idx = [int(np.argmax([tot[p] for p in players]))]
nbrs = {}
for q in q_idx:
    order = [j for j in np.argsort(Dm[q]) if j != q][:TOP_N]
    nbrs[q] = order

# ---------- reliability by re-sampling each player's matches ----------
aris, keep_hits = [], {q: np.zeros(TOP_N) for q in q_idx}
for b in range(N_BOOT):
    Sb = np.vstack([CNT[rng.choice(rows_of[p], len(rows_of[p]), replace=True)].sum(axis=0) for p in players])
    Zb = zscore(features_from(Sb))
    lb = kmeans(Zb, K, rng, n_init=5)[1]
    aris.append(ari(labels0, lb))
    Db = np.sqrt(((Zb[:, None] - Zb[None]) ** 2).sum(2))
    for q in q_idx:
        top_b = set([j for j in np.argsort(Db[q]) if j != q][:TOP_N])
        for r, j in enumerate(nbrs[q]):
            keep_hits[q][r] += (j in top_b)
aris = np.array(aris)
for q in q_idx:
    keep_hits[q] /= N_BOOT
print(f"\nRe-sampling each player's matches {N_BOOT} times: agreement of the roles with the original "
      f"(adjusted Rand index) mean {aris.mean():.2f}, 5th to 95th {np.percentile(aris, 5):.2f} to {np.percentile(aris, 95):.2f}")

print(f"\nWho plays like whom (closest {TOP_N} by distance in the 12-feature space):")
nb_rows = []
for q in q_idx:
    p = players[q]
    print(f"\n  {names[p]} ({codes[q]}, {pg[q]}, role {labels0[q] + 1}, {tot[p]:.0f} actions, {n_matches[q]} matches)  [{p}]")
    for r, j in enumerate(nbrs[q]):
        print(f"     {r + 1}. {names[players[j]]:<20}{codes[j]}  {pg[j]:<14} role {labels0[j] + 1}  "
              f"distance {Dm[q, j]:.2f}  stays in top {TOP_N}: {keep_hits[q][r] * 100:.0f}%  [{players[j]}]")
        nb_rows.append(dict(query=p, rank=r + 1, neighbour=players[j], team=team_of[players[j]],
                            listed_position=pg[j], role=int(labels0[j] + 1), distance=float(Dm[q, j]),
                            stays_in_top5=float(keep_hits[q][r])))
    same_team = sum(1 for j in nbrs[q] if team_of[players[j]] == team_of[p])
    print(f"     {same_team} of {TOP_N} neighbours are team-mates")
pd.DataFrame(nb_rows).to_csv("player_neighbours.csv", index=False)

out = pd.DataFrame(X0, columns=FEATS)
out.insert(0, "player", players)
out.insert(1, "name", [names[p] for p in players])
out.insert(2, "team", [team_of[p] for p in players])
out.insert(3, "matches", n_matches)
out.insert(4, "actions", [tot[p] for p in players])
out.insert(5, "listed_position", pg)
out["role"] = labels0 + 1
out["pc1"], out["pc2"] = scores[:, 0], scores[:, 1]
out.to_csv("player_roles.csv", index=False)

# ---------- label placement ----------
def place_labels(fig, ax, xs, ys, labels, sizes, bold=None):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    box = fig.bbox
    placed, result = [], {}
    crowd = [sum(np.hypot(*(pts[i] - pts[j])) < 70 for j in range(len(xs)) if j != i) for i in range(len(xs))]
    for i in sorted(range(len(xs)), key=lambda k: -crowd[k]):
        t = ax.text(0, 0, labels[i], fontsize=sizes[i], fontweight="bold" if (bold and bold[i]) else "normal")
        bb = t.get_window_extent(rend)
        w, h = bb.width, bb.height
        t.remove()
        cands = [(-w / 2, 9), (-w / 2, -9 - h), (9, -h / 2), (-9 - w, -h / 2),
                 (8, 7), (-8 - w, 7), (8, -7 - h), (-8 - w, -7 - h),
                 (-w / 2, 20), (-w / 2, -20 - h), (20, -h / 2), (-20 - w, -h / 2)]
        best = None
        for cj, (dx, dy) in enumerate(cands):
            x0, y0 = pts[i][0] + dx, pts[i][1] + dy
            pen = cj * 0.5
            if x0 < box.x0 + 2 or x0 + w > box.x1 - 2 or y0 < box.y0 + 2 or y0 + h > box.y1 - 2:
                pen += 200
            for (a0, b0, a1, b1) in placed:
                ox = max(0.0, min(x0 + w, a1) - max(x0, a0))
                oy = max(0.0, min(y0 + h, b1) - max(y0, b0))
                pen += 3 * ox * oy
            for j in range(len(xs)):
                px_, py_ = pts[j]
                ox = max(0.0, min(x0 + w, px_ + 7) - max(x0, px_ - 7))
                oy = max(0.0, min(y0 + h, py_ + 7) - max(y0, py_ - 7))
                pen += 2 * ox * oy
            if best is None or pen < best[0]:
                best = (pen, dx, dy, x0, y0, w, h)
        _, dx, dy, x0, y0, w, h = best
        placed.append((x0, y0, x0 + w, y0 + h))
        result[i] = (dx, dy)
    return result


xr = scores[:, 0].max() - scores[:, 0].min()
yr = scores[:, 1].max() - scores[:, 1].min()
XLIM = (scores[:, 0].min() - 0.08 * xr, scores[:, 0].max() + 0.08 * xr)
YLIM = (scores[:, 1].min() - 0.10 * yr, scores[:, 1].max() + 0.10 * yr)
role_col = [ROLE_COLORS[c] for c in labels0]

# ---------- chart 1: the role map ----------
fig1 = plt.figure(figsize=(11, 9.5), dpi=150)
fig1.patch.set_facecolor(BG)
ax1 = fig1.add_axes([0.07, 0.39, 0.89, 0.50])
ax1.set_facecolor(BG)
ax1.set_xlim(*XLIM)
ax1.set_ylim(*YLIM)
ax1.scatter(scores[:, 0], scores[:, 1], s=46, c=role_col, alpha=0.8, edgecolors="none", zorder=3)
ax1.scatter(scores[q_idx, 0], scores[q_idx, 1], s=110, c=[role_col[i] for i in q_idx], edgecolors="white",
            linewidths=1.6, zorder=5)
ax1.axhline(0, color=GRID, linewidth=0.6)
ax1.axvline(0, color=GRID, linewidth=0.6)
ax1.set_xlabel(f"PC1 ({expl[0] * 100:.0f}% of the variance in the 12 features)", color="white", fontsize=10.5)
ax1.set_ylabel(f"PC2 ({expl[1] * 100:.0f}%)", color="white", fontsize=10.5)
ax1.tick_params(colors="white", labelsize=9)
for sp in ax1.spines.values():
    sp.set_color(GRID)
lab_x = [scores[i, 0] for i in q_idx]
lab_y = [scores[i, 1] for i in q_idx]
lab_t = [names[players[i]] for i in q_idx]
offs1 = place_labels(fig1, ax1, lab_x, lab_y, lab_t, [10.5] * len(q_idx), [True] * len(q_idx))
for k_, i in enumerate(q_idx):
    ax1.annotate(lab_t[k_], xy=(lab_x[k_], lab_y[k_]), xytext=offs1[k_], textcoords="offset pixels", ha="left",
                 va="bottom", color="white", fontsize=10.5, fontweight="bold", zorder=7)
fig1.text(0.5, 0.955, "Player Role Map: What Copa America 2024 Players Do on the Ball", ha="center", color="white",
          fontsize=16.5, fontweight="bold")
fig1.text(0.5, 0.92, f"{nP} outfield players with {MIN_ACTIONS}+ actions, 12 on-ball features, k-means with k = {K} "
          f"(silhouette {best_sil:.2f}). Argentina players labelled", ha="center", color=MUTED, fontsize=10)
yy = 0.30
fig1.text(0.04, yy, "Roles (numbered from deepest to highest position, the colours carry no name)", color="white",
          fontsize=10.5, fontweight="bold")
yy -= 0.035
step = min(0.034, 0.22 / max(K, 1))
for c in range(K):
    n_c, comp_txt, hi, lo = role_desc[c]
    fig1.text(0.04, yy, f"Role {c + 1} ({n_c})", color=ROLE_COLORS[c], fontsize=9.5, fontweight="bold")
    fig1.text(0.17, yy, f"listed as: {comp_txt}.  higher: {', '.join(hi) or 'none'};  lower: {', '.join(lo) or 'none'}",
              color=MUTED, fontsize=9)
    yy -= step
fig1.text(0.5, 0.012, "A 2D view of 12 features, so distances on the map are approximate. Shares of actions, not per-90 rates.",
          ha="center", color=MUTED, fontsize=9)
fig1.savefig("player_role_map.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: player_role_map.png")

# ---------- chart 2: roles vs positions, and a lookup ----------
fig2 = plt.figure(figsize=(12, 7.2), dpi=150)
fig2.patch.set_facecolor(BG)
axh = fig2.add_axes([0.07, 0.17, 0.36, 0.65])
cmap = LinearSegmentedColormap.from_list("rp", ["#10253a", BLUE, "#ffffff"])
im = axh.imshow(cont, cmap=cmap, aspect="auto", vmin=0, vmax=max(cont.max(), 1))
axh.set_xticks(range(len(POS_GROUPS)))
axh.set_xticklabels(POS_GROUPS, rotation=35, ha="right", color="white", fontsize=9)
axh.set_yticks(range(K))
axh.set_yticklabels([f"Role {c + 1}" for c in range(K)], color="white", fontsize=9.5)
for c in range(K):
    for j in range(len(POS_GROUPS)):
        if cont[c, j] > 0:
            axh.text(j, c, str(cont[c, j]), ha="center", va="center", fontsize=9,
                     color="#0d1b2a" if cont[c, j] > cont.max() * 0.55 else "white")
axh.tick_params(length=0)
for sp in axh.spines.values():
    sp.set_color(GRID)
axh.set_title("Do the roles rediscover listed positions?", color="white", fontsize=11, pad=8)
q0 = q_idx[0]
axb = fig2.add_axes([0.67, 0.17, 0.30, 0.65])
axb.set_facecolor(BG)
nb = nbrs[q0]
ds = [Dm[q0, j] for j in nb][::-1]
lbl = [f"{names[players[j]]} ({codes[j]})" for j in nb][::-1]
stay_ = keep_hits[q0][::-1]
axb.barh(range(len(nb)), ds, color=ORANGE, height=0.6)
for r, (d_, s_) in enumerate(zip(ds, stay_)):
    axb.text(d_ + max(ds) * 0.02, r, f"{d_:.2f}   stays {s_ * 100:.0f}%", va="center", color="white", fontsize=9.5)
axb.set_yticks(range(len(nb)))
axb.set_yticklabels(lbl, color="white", fontsize=9.5)
axb.set_xlim(0, max(ds) * 1.55)
axb.tick_params(axis="x", colors="white", labelsize=9)
axb.tick_params(axis="y", length=0)
axb.set_xlabel("Distance in the 12-feature space (smaller = more alike)", color="white", fontsize=9.5)
for sp in axb.spines.values():
    sp.set_color(GRID)
axb.set_title(f"Who plays most like {names[players[q0]]}?", color="white", fontsize=11, pad=8)
fig2.suptitle("Checking the Roles and the Lookup", color="white", fontsize=17, fontweight="bold", y=0.975)
fig2.text(0.5, 0.905, f"Roles vs listed positions: adjusted Rand index {ari_pos:.2f}, purity {purity * 100:.0f}%. "
          f"'Stays' = share of {N_BOOT} re-samples in which the neighbour is still in the top {TOP_N}",
          ha="center", color=MUTED, fontsize=10)
fig2.text(0.5, 0.035, f"Roles re-sampled {N_BOOT} times: agreement with the original (ARI) mean {aris.mean():.2f}, "
          f"5th to 95th percentile {np.percentile(aris, 5):.2f} to {np.percentile(aris, 95):.2f}",
          ha="center", color="white", fontsize=10.5)
fig2.savefig("player_role_lookup.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: player_role_lookup.png")

# ---------- replay: the lookup, one player at a time ----------
vq = q_idx[:VIDEO_QUERIES]
fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axm = fig.add_axes([0.05, 0.40, 0.90, 0.42])
fig.text(0.5, 0.965, "Who Plays Like Who?", ha="center", color="white", fontsize=25, fontweight="bold")
fig.text(0.5, 0.928, f"Copa America 2024, {nP} outfield players, closest by on-ball action mix", ha="center",
         color=MUTED, fontsize=14)
info_t = fig.text(0.5, 0.885, "", ha="center", color="white", fontsize=18, fontweight="bold")
list_t = [fig.text(0.12, 0.345 - 0.047 * r, "", ha="left", color="white", fontsize=15) for r in range(TOP_N)]
rel_t = fig.text(0.5, 0.075, "", ha="center", color=GREEN, fontsize=15)
fig.text(0.5, 0.012, "A 2D view of 12 features, so neighbours can look further apart than they are. Similar action mix, not similar quality.",
         ha="center", color=MUTED, fontsize=8.5)
axm.set_xlim(*XLIM)
axm.set_ylim(*YLIM)
offs_v = {}
for q in vq:
    ids = [q] + list(nbrs[q])
    offs_v[q] = place_labels(fig, axm, [scores[i, 0] for i in ids], [scores[i, 1] for i in ids],
                             [names[players[i]] for i in ids], [11] * len(ids), [True] * len(ids))


def draw_state(st):
    axm.clear()
    axm.set_facecolor(BG)
    axm.set_xlim(*XLIM)
    axm.set_ylim(*YLIM)
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color(GRID)
    q = st.get("q")
    axm.scatter(scores[:, 0], scores[:, 1], s=60, c=role_col, alpha=0.35 if q is not None else 0.7,
                edgecolors="none", zorder=2)
    for t_ in list_t:
        t_.set_text("")
    rel_t.set_text("")
    if q is None:
        info_t.set_text(st["info"])
        return
    shown = st["shown"]
    ids = [q] + list(nbrs[q][:shown])
    for j in nbrs[q][:shown]:
        axm.plot([scores[q, 0], scores[j, 0]], [scores[q, 1], scores[j, 1]], color=ORANGE, linewidth=1.4, alpha=0.7, zorder=3)
        axm.scatter([scores[j, 0]], [scores[j, 1]], s=170, c=[role_col[j]], edgecolors=ORANGE, linewidths=2.6, zorder=5)
    axm.scatter([scores[q, 0]], [scores[q, 1]], s=260, c=[role_col[q]], edgecolors="white", linewidths=3, zorder=6)
    for k_, i in enumerate(ids):
        axm.annotate(names[players[i]], xy=(scores[i, 0], scores[i, 1]), xytext=offs_v[q][k_],
                     textcoords="offset pixels", ha="left", va="bottom", color="white" if i == q else ORANGE,
                     fontsize=11, fontweight="bold", zorder=8)
    info_t.set_text(f"{names[players[q]]} ({codes[q]}), listed as {pg[q].lower()}")
    for r, j in enumerate(nbrs[q][:shown]):
        list_t[r].set_text(f"{r + 1}.  {names[players[j]]} ({codes[j]}), {pg[j].lower()},  distance {Dm[q, j]:.2f}")
    if st.get("rel"):
        avg = keep_hits[q].mean() * 100
        rel_t.set_text(f"Re-sample the matches and a neighbour stays in the top {TOP_N} {avg:.0f}% of the time")
        rel_t.set_color(GREEN if avg >= 70 else (ORANGE if avg >= 40 else RED))


states = [dict(q=None, info="Every player is a point, coloured by role") for _ in range(16)]
for q in vq:
    for _ in range(6):
        states.append(dict(q=q, shown=0))
    for s in range(1, TOP_N + 1):
        for _ in range(6):
            states.append(dict(q=q, shown=s))
    for _ in range(14):
        states.append(dict(q=q, shown=TOP_N, rel=True))
for _ in range(30):
    states.append(dict(q=None, info=f"Roles vs listed positions: ARI {ari_pos:.2f}, purity {purity * 100:.0f}%"))

FPS = 20
RAW_VIDEO = "player_role_replay_raw.mp4"
FINAL_VIDEO = "player_role_replay.mp4"
anim = FuncAnimation(fig, lambda i: draw_state(states[i]), frames=len(states), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(states)} frames ({len(states) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm_f = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
              f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
              f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts_f = [f"[{i}:v]{norm_f}[v{i}]" for i in range(n)]
    labels_f = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts_f) + f";{labels_f}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)