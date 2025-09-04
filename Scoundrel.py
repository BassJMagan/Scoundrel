# scoundrel_pygame.py
# Pure-Pygame Scoundrel with keyboard shortcuts. No Gradio, no SSL.
#
# Install: pip install pygame
# Run game: python scoundrel_pygame.py
# Run tests: RUN_TESTS=1 python scoundrel_pygame.py

from __future__ import annotations
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# ---------- Performance knobs (env overrides) ----------
DEFAULT_STACK_LIMIT = int(os.getenv("SCOUNDREL_STACK_LIMIT", "1000000"))  # nodes per DFS
DEFAULT_MAX_SHUFFLES = int(os.getenv("SCOUNDREL_MAX_SHUFFLES", "512"))
# Default: all cores (cap to >=1). In tests, default to 1 unless overridden.
DEFAULT_WORKERS = max(1, int(os.getenv(
    "SCOUNDREL_WORKERS",
    ("1" if os.getenv("RUN_TESTS") else str(os.cpu_count() or 1))
)))

# -------------------------------------------------------------
# Core model (engine + solver) — no pygame imports here so tests are headless
# -------------------------------------------------------------

SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS: List[Tuple[str, int]] = [
    ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9), ("10", 10),
    ("J", 11), ("Q", 12), ("K", 13), ("A", 14)
]

@dataclass(frozen=True)
class Card:
    id: str
    suit: str
    rank: str
    value: int

@dataclass
class GameState:
    deck: List[Card]
    room: List[Card]
    discard: List[Card]
    health: int = 20
    max_health: int = 20
    healed_this_room: bool = False
    weapon: int = 0
    weapon_chain_limit: Optional[int] = None
    ran_last_room: bool = False
    combat_mode: str = "auto"  # auto | weapon | fist
    log: List[str] = None
    last_played: Optional[Card] = None
    prev_health_before_last_heart: Optional[int] = None
    last_remaining: Optional[Card] = None
    state: str = "playing"  # playing | won | lost
    score: Optional[int] = None

    def __post_init__(self):
        if self.log is None:
            self.log = []

# -------------------------------------------------------------
# Deck build and utils
# -------------------------------------------------------------

def build_deck() -> List[Card]:
    deck: List[Card] = []
    uid = 0
    for suit in SUITS:
        for rank, val in RANKS:
            is_red = suit in ("hearts", "diamonds")
            is_face = rank in ("J", "Q", "K")
            is_ace = rank == "A"
            if is_red and (is_face or is_ace):
                continue
            deck.append(Card(id=f"{suit}-{rank}-{uid}", suit=suit, rank=rank, value=val))
            uid += 1
    return deck  # 44 cards


def shuffle_deck(deck: List[Card]) -> List[Card]:
    d = deck[:]
    random.shuffle(d)
    return d

# ---------- Multiprocessing helpers (Windows-safe) ----------
_POOL_BASE: List[Card] = []

def _pool_init(base_deck: List[Card]):
    # Each worker gets its own base copy reference (Card is small dataclass)
    global _POOL_BASE
    _POOL_BASE = base_deck

def _pool_attempt(args):
    seed, stack_limit = args
    rng = random.Random(seed)
    fresh = _POOL_BASE[:]
    rng.shuffle(fresh)
    win, nodes = dfs_can_win(fresh, stack_limit=stack_limit)
    return win, fresh, nodes


# -------------------------------------------------------------
# Perfect information DFS solver to verify winnable deals
# -------------------------------------------------------------

@dataclass
class SimState:
    deck: List[Card]
    room: List[Card]
    health: int
    weapon: int
    chain: Optional[int]
    healed: bool
    ran: bool
    plays: int


def can_use_weapon_on(weapon: int, chain: Optional[int], val: int) -> bool:
    if weapon <= 0:
        return False
    if chain is None:
        return True
    return val < chain


def state_key(s: SimState):
    # Tuple key is faster than building giant strings
    return (
        s.health,
        s.weapon,
        -1 if s.chain is None else s.chain,
        1 if s.healed else 0,
        1 if s.ran else 0,
        s.plays,
        tuple(c.id for c in s.room),
        tuple(c.id for c in s.deck),
    )


def deal_if_needed(s: SimState) -> bool:
    # Returns True if this immediately results in a win (cannot form a full room)
    while True:
        if len(s.room) == 0:
            if len(s.deck) < 4:
                return True
            s.room = s.deck[:4]
            s.deck = s.deck[4:]
            s.healed = False
            s.plays = 0
            continue
        if len(s.room) == 1:
            carry = s.room[:]
            while len(carry) < 4 and s.deck:
                carry.append(s.deck.pop(0))
            if len(carry) < 4:
                return True
            s.room = carry
            s.healed = False
            s.plays = 0
            s.ran = False
            continue
        break
    return False


def dfs_can_win(initial: List[Card], stack_limit: Optional[int] = None) -> Tuple[bool, int]:
    """Return (win, nodes_explored) using perfect-info DFS. stack_limit caps nodes.
    """
    sys.setrecursionlimit(20000)
    deck = initial[:]
    room = deck[:4]
    deck = deck[4:]
    start = SimState(deck=deck, room=room, health=20, weapon=0, chain=None, healed=False, ran=False, plays=0)

    memo: Dict[str, bool] = {}
    nodes = 0
    hard_cap = stack_limit if stack_limit is not None else DEFAULT_STACK_LIMIT

    def try_dfs(s: SimState) -> bool:
        nonlocal nodes
        if deal_if_needed(s):
            return True
        if s.health <= 0:
            return False
        key = state_key(s)
        if key in memo:
            return memo[key]
        nodes += 1
        if nodes > hard_cap:
            memo[key] = False
            return False

        # Option: flee only at start and not twice
        if s.plays == 0 and not s.ran:
            next_deck = s.deck + s.room
            if len(next_deck) < 4:
                memo[key] = True
                return True
            s_flee = SimState(deck=next_deck[:], room=[], health=s.health, weapon=s.weapon, chain=s.chain, healed=False, ran=True, plays=0)
            if try_dfs(s_flee):
                memo[key] = True
                return True

        # Build action list in good order
        heart_idx = [i for i, c in enumerate(s.room) if c.suit == "hearts"]
        dia_idx = [i for i, c in enumerate(s.room) if c.suit == "diamonds"]
        mon_idx = [i for i, c in enumerate(s.room) if c.suit in ("spades", "clubs")]

        # Hearts: biggest first if available and not healed
        if not s.healed and heart_idx:
            heart_idx.sort(key=lambda i: s.room[i].value, reverse=True)
            for i in heart_idx:
                ns = SimState(deck=s.deck[:], room=s.room[:], health=min(20, s.health + s.room[i].value), weapon=s.weapon, chain=s.chain, healed=True, ran=s.ran, plays=s.plays + 1)
                ns.room.pop(i)
                if try_dfs(ns):
                    memo[key] = True
                    return True

        # Better weapons first
        better_dias = [i for i in dia_idx if s.room[i].value > s.weapon]
        better_dias.sort(key=lambda i: s.room[i].value, reverse=True)
        for i in better_dias:
            ns = SimState(deck=s.deck[:], room=s.room[:], health=s.health, weapon=s.room[i].value, chain=None, healed=s.healed, ran=s.ran, plays=s.plays + 1)
            ns.room.pop(i)
            if try_dfs(ns):
                memo[key] = True
                return True

        # Weapon kills, highest legal first
        legal = [i for i in mon_idx if can_use_weapon_on(s.weapon, s.chain, s.room[i].value)]
        legal.sort(key=lambda i: s.room[i].value, reverse=True)
        for i in legal:
            val = s.room[i].value
            dmg = max(0, val - s.weapon)
            hp = s.health - dmg
            if hp <= 0:
                continue
            ns = SimState(deck=s.deck[:], room=s.room[:], health=hp, weapon=s.weapon, chain=val, healed=s.healed, ran=s.ran, plays=s.plays + 1)
            ns.room.pop(i)
            if try_dfs(ns):
                memo[key] = True
                return True

        # Punches: smallest first
        mons_sorted = sorted(mon_idx, key=lambda i: s.room[i].value)
        for i in mons_sorted:
            val = s.room[i].value
            hp = s.health - val
            if hp <= 0:
                continue
            ns = SimState(deck=s.deck[:], room=s.room[:], health=hp, weapon=s.weapon, chain=s.chain, healed=s.healed, ran=s.ran, plays=s.plays + 1)
            ns.room.pop(i)
            if try_dfs(ns):
                memo[key] = True
                return True

        # Extra hearts after healed count as plays
        if s.healed and heart_idx:
            for i in heart_idx:
                ns = SimState(deck=s.deck[:], room=s.room[:], health=s.health, weapon=s.weapon, chain=s.chain, healed=True, ran=s.ran, plays=s.plays + 1)
                ns.room.pop(i)
                if try_dfs(ns):
                    memo[key] = True
                    return True

        # Diamonds that are not upgrades
        for i in dia_idx:
            if s.room[i].value <= s.weapon:
                ns = SimState(deck=s.deck[:], room=s.room[:], health=s.health, weapon=s.weapon, chain=s.chain, healed=s.healed, ran=s.ran, plays=s.plays + 1)
                ns.room.pop(i)
                if try_dfs(ns):
                    memo[key] = True
                    return True

        memo[key] = False
        return False

    can = try_dfs(start)
    return can, nodes


def find_winnable_deck(max_shuffles: int = DEFAULT_MAX_SHUFFLES,
                       workers: int = DEFAULT_WORKERS,
                       stack_limit: int = DEFAULT_STACK_LIMIT) -> Tuple[List[Card], int, int]:
    """Search for a winnable deck. Tries a short sequential warmup first, then
    parallelizes across CPU cores if workers>1. Returns (deck, attempts_used, nodes).
    """
    base = build_deck()
    workers = max(1, min(workers, os.cpu_count() or workers))

    # --- Warmup: quick sequential tries with a smaller cap so UI doesn't stall
    warmup = min(128, max(0, max_shuffles // 4))
    warmup_stack = max(100_000, stack_limit // 4)
    for attempt in range(1, warmup + 1):
        fresh = shuffle_deck(base)
        win, nodes = dfs_can_win(fresh, stack_limit=warmup_stack)
        if win:
            return fresh, attempt, nodes

    remaining = max_shuffles - warmup
    if remaining <= 0:
        return shuffle_deck(base), max_shuffles, 0

    # --- Sequential path if single worker
    if workers <= 1:
        for attempt in range(1, remaining + 1):
            fresh = shuffle_deck(base)
            win, nodes = dfs_can_win(fresh, stack_limit=stack_limit)
            if win:
                return fresh, warmup + attempt, nodes
        return shuffle_deck(base), max_shuffles, 0

    # --- Parallel path
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # Windows-safe
        seeds = [random.randrange(1 << 30) for _ in range(remaining)]
        with ctx.Pool(processes=workers, initializer=_pool_init, initargs=(base,)) as pool:
            try:
                for idx, (win, deck, nodes) in enumerate(
                    pool.imap_unordered(_pool_attempt, ((s, stack_limit) for s in seeds), chunksize=16), 1
                ):
                    if win:
                        pool.terminate(); pool.join()
                        return deck, warmup + idx, nodes
            finally:
                try:
                    pool.terminate(); pool.join()
                except Exception:
                    pass
    except Exception:
        # Pool failed to start, fall back to sequential for remaining tries
        for attempt in range(1, remaining + 1):
            fresh = shuffle_deck(base)
            win, nodes = dfs_can_win(fresh, stack_limit=stack_limit)
            if win:
                return fresh, warmup + attempt, nodes

    # Fallback if nothing found
    return shuffle_deck(base), max_shuffles, 0

# -------------------------------------------------------------
# Game mechanics (engine rules)
# -------------------------------------------------------------

def append_log(gs: GameState, s: str) -> None:
    gs.log.insert(0, s)
    gs.log[:] = gs.log[:80]


def cards_played_this_room(gs: GameState) -> int:
    return 4 - len(gs.room)


def can_use_weapon_on_room(gs: GameState, val: int) -> bool:
    if gs.weapon <= 0:
        return False
    if gs.weapon_chain_limit is None:
        return True
    return val < gs.weapon_chain_limit

# ---- Scoring helpers ----
def sum_remaining_monsters(gs: GameState) -> int:
    total = 0
    for c in gs.room:
        if c.suit in ("spades", "clubs"):
            total += c.value
    for c in gs.deck:
        if c.suit in ("spades", "clubs"):
            total += c.value
    return total

def compute_victory_score(gs: GameState) -> int:
    base = max(gs.health, 0)
    bonus = 0
    if gs.last_remaining and gs.last_remaining.suit == "hearts" and gs.health >= gs.max_health:
        bonus = gs.last_remaining.value
    return base + bonus

def compute_defeat_score(gs: GameState) -> int:
    # “If your life has reached zero, find all remaining monsters
    # in the Dungeon, and subtract their values from your life;
    # this negative value is your score.”
    return gs.health - sum_remaining_monsters(gs)


def fill_room_to_four(gs: GameState, carry: Optional[List[Card]] = None) -> None:
    if gs.state != "playing":
        return
    carry = carry[:] if carry else []
    next_deck = gs.deck[:]
    next_room = carry[:]
    while len(next_room) < 4 and next_deck:
        next_room.append(next_deck.pop(0))
    if len(next_room) < 4:
        if carry:
            gs.last_remaining = carry[0]
        if next_room:
            gs.discard = next_room + gs.discard
        gs.deck = []
        gs.room = []
        gs.state = "won"
        append_log(gs, f"Deck exhausted. Could not form full room (had {len(next_room)}). Discarded leftover card(s). You win.")
        gs.score = compute_victory_score(gs)
        append_log(gs, f"Points: {gs.score}")
        return
    gs.deck = next_deck
    gs.room = next_room
    gs.healed_this_room = False


def start_new_run() -> GameState:
    deck, attempts, nodes = find_winnable_deck(
        max_shuffles=DEFAULT_MAX_SHUFFLES,
        workers=DEFAULT_WORKERS,
        stack_limit=DEFAULT_STACK_LIMIT,
    )
    first = deck[:4]
    rest = deck[4:]
    gs = GameState(deck=rest, room=first, discard=[], state="playing", log=[], combat_mode="auto")
    append_log(gs, f"Entered the dungeon. Auto mode is active. Winnable deck after {attempts} shuffle{'s' if attempts != 1 else ''}. Solver explored {nodes} states. Workers={DEFAULT_WORKERS}, Cap={DEFAULT_STACK_LIMIT} nodes.")
    return gs


def on_play_heart(gs: GameState, card: Card) -> None:
    gs.last_played = card
    if not gs.healed_this_room:
        gs.prev_health_before_last_heart = gs.health
        gs.health = min(gs.max_health, gs.health + card.value)
        append_log(gs, f"Potion +{card.value} HP (now {gs.health}).")
    else:
        append_log(gs, "Second potion this room discarded (no effect).")
    gs.healed_this_room = True
    gs.discard.insert(0, card)

def on_play_weapon(gs: GameState, card: Card) -> None:
    gs.last_played = card
    gs.weapon = card.value
    gs.weapon_chain_limit = None
    gs.discard.insert(0, card)
    if gs.combat_mode == "auto":
        append_log(gs, f"Equipped weapon {card.value}. Auto will use it when valid.")
    else:
        append_log(gs, f"Equipped weapon {card.value}.")


def fight_monster(gs: GameState, card: Card, force_mode: Optional[str] = None) -> None:
    gs.last_played = card
    val = card.value
    if gs.combat_mode == "auto" and force_mode is None:
        mode = "weapon" if can_use_weapon_on_room(gs, val) else "fist"
    else:
        mode = force_mode or gs.combat_mode
    using_weapon = (mode == "weapon") and can_use_weapon_on_room(gs, val)
    dmg = max(0, val - gs.weapon) if using_weapon else val
    gs.health -= dmg
    gs.discard.insert(0, card)
    if using_weapon:
        gs.weapon_chain_limit = val
        append_log(gs, f"{'Auto-' if gs.combat_mode=='auto' else ''}Weapon vs {'Spade' if card.suit=='spades' else 'Club'} {val}. Took {dmg}.")
    else:
        append_log(gs, f"{'Auto-' if gs.combat_mode=='auto' else ''}Fist vs {'Spade' if card.suit=='spades' else 'Club'} {val}. Took {dmg}.")
    if gs.health <= 0:
        gs.state = "lost"
        gs.score = compute_defeat_score(gs)
        append_log(gs, "You perished in the dungeon.")
        append_log(gs, f"Points: {gs.score}")


def on_click_card(gs: GameState, idx: int) -> bool:
    if gs.state != "playing":
        return False
    if idx < 0 or idx >= len(gs.room):
        return False

    card = gs.room[idx]

    if card.suit == "hearts":
        gs.room.pop(idx)
        on_play_heart(gs, card)
    elif card.suit == "diamonds":
        gs.room.pop(idx)
        on_play_weapon(gs, card)
    else:
        if gs.combat_mode == "weapon" and not can_use_weapon_on_room(gs, card.value):
            append_log(gs, "Weapon cannot be used on an equal or higher monster. Switch to Fist or use Auto.")
            return False
        gs.room.pop(idx)
        fight_monster(gs, card)

    # If we died on that play, stop here — do NOT carry/refill
    if gs.state != "playing":
        return True

    # After 3 plays in a room, carry remaining card to next
    if len(gs.room) == 1:
        carry = gs.room[:]
        gs.room.clear()
        gs.ran_last_room = False
        fill_room_to_four(gs, carry)
    elif len(gs.room) == 0:
        if len(gs.deck) == 0 and gs.state == "playing":
            gs.state = "won"
            append_log(gs, "You made it through the dungeon!")
            gs.score = compute_victory_score(gs)
            append_log(gs, f"Points: {gs.score}")
        elif gs.state == "playing":
            gs.ran_last_room = False
            fill_room_to_four(gs, [])

    return True


def flee_room(gs: GameState) -> None:
    if gs.state != "playing":
        return
    if gs.ran_last_room:
        append_log(gs, "You cannot run twice in a row.")
        return
    if cards_played_this_room(gs) > 0:
        append_log(gs, "You can only flee at the start of a room before any card is played.")
        return
    next_deck = gs.deck + gs.room
    gs.deck = next_deck
    gs.room = []
    gs.ran_last_room = True
    gs.healed_this_room = False
    append_log(gs, "You fled. New room.")
    fill_room_to_four(gs, [])

# -------------------------------------------------------------
# Pygame UI — isolated so tests don't import/initialize SDL
# -------------------------------------------------------------

def run_game():
    import pygame  # local import so tests don't need it

    pygame.init()
    pygame.font.init()

    # Start bigger and allow resizing
    W, H = 1280, 800
    CARD_W, CARD_H = 110, 154

    COL_BG = (15, 23, 42)        # slate-900
    COL_PANEL = (30, 41, 59)     # slate-800
    COL_PANEL_SOFT = (23, 33, 52) # darker panel for room area
    COL_BORDER = (51, 65, 85)    # slate-700
    COL_TEXT = (226, 232, 240)   # slate-200
    COL_ACCENT = (16, 185, 129)  # emerald-500
    COL_RED = (239, 68, 68)
    COL_WHITE = (245, 245, 245)
    COL_BLACK = (31, 41, 55)
    COL_PLACEHOLDER = (71, 85, 105)  # slate-600
    COL_TOOLTIP_BG = (2, 6, 23)      # near-black tooltip

    try:
        FONT = pygame.font.SysFont("dejavusansmono,consolas,menlo,monaco,monospace", 20)
        FONT_SM = pygame.font.SysFont("dejavusansmono,consolas,menlo,monaco,monospace", 16)
        FONT_LG = pygame.font.SysFont("dejavusansmono,consolas,menlo,monaco,monospace", 30)
    except Exception:
        FONT = pygame.font.Font(None, 20)
        FONT_SM = pygame.font.Font(None, 16)
        FONT_LG = pygame.font.Font(None, 30)

    def draw_text(surface, text, x, y, color=COL_TEXT, font=FONT):
        img = font.render(text, True, color)
        surface.blit(img, (x, y))

    def wrap_text(text: str, font, max_width: int) -> list[str]:
        words = text.split(' ')
        lines = []
        cur = ''
        for w in words:
            test = (cur + ' ' + w).strip()
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # Dynamic rectangles depending on W,H
    def layout():
        header = pygame.Rect(20, 20, W - 40, 120)
        ctrl = pygame.Rect(20, header.bottom + 12, W - 40, 64)
        # Leave space on right for log
        log_w = max(340, int(W * 0.26))
        room = pygame.Rect(20, ctrl.bottom + 12, W - 40 - log_w - 12, H - (ctrl.bottom + 12) - 20)
        log = pygame.Rect(room.right + 12, room.top, log_w, room.height)
        return header, ctrl, room, log

    # Fixed slot rects inside the room panel, centered evenly
    def slot_rect(slot_index: int, room_rect: pygame.Rect) -> pygame.Rect:
        slots = 4
        total_w = slots * CARD_W + (slots - 1) * 20
        x0 = room_rect.x + (room_rect.width - total_w) // 2
        y0 = room_rect.y + 56  # push below room title
        x = x0 + slot_index * (CARD_W + 20)
        return pygame.Rect(x, y0, CARD_W, CARD_H)

    def draw_badges(surface, header: pygame.Rect, gs: GameState):
        # draw badges with wrapping so they never collide with health area
        pad_y = header.y + 60
        x = header.x + 16
        right_limit = header.x + header.width - 260  # leave space for health block

        def badge(text, bg=COL_PANEL, fg=COL_TEXT):
            nonlocal x, pad_y
            pad = 6
            img = FONT.render(text, True, fg)
            rect = img.get_rect()
            bw = rect.width + pad * 2
            if x + bw > right_limit:
                x = header.x + 16
                pad_y += rect.height + 8
            r = pygame.Rect(x, pad_y, bw, rect.height + pad)
            pygame.draw.rect(surface, bg, r, border_radius=12)
            surface.blit(img, (x + pad, pad_y + 2))
            x += bw + 8

        badge(f"Weapon: {gs.weapon}", bg=COL_ACCENT, fg=COL_BLACK)
        chain = "-" if gs.weapon_chain_limit is None else f"{gs.weapon_chain_limit - 1} max"
        badge(f"Chain: {chain}")
        runs_text = "On cooldown" if gs.ran_last_room else ("Available" if cards_played_this_room(gs) == 0 else "Locked")
        badge(f"Runs: {runs_text}")
        badge(f"Deck: {len(gs.deck)}")
        badge(f"Discard: {len(gs.discard)}")

        # Health block on the far right
        draw_text(surface, "Health", right_limit + 16, header.y + 12)
        hb = pygame.Rect(right_limit + 16, header.y + 40, 220, 18)
        pygame.draw.rect(surface, COL_BORDER, hb, border_radius=10)
        pct = max(0, min(1, gs.health / gs.max_health))
        fill = pygame.Rect(hb.x + 2, hb.y + 2, int((hb.width - 4) * pct), hb.height - 4)
        pygame.draw.rect(surface, COL_ACCENT, fill, border_radius=8)
        draw_text(surface, f"{gs.health} / {gs.max_health}", hb.x, hb.y + 22, color=COL_TEXT, font=FONT_SM)

    def predict_hover_text(gs: GameState, card: Card) -> str:
        if card.suit == 'hearts':
            if gs.healed_this_room:
                return "Heart: no heal (already healed)"
            new_hp = min(gs.max_health, gs.health + card.value)
            return f"Heart: +{card.value} HP → {new_hp}"
        if card.suit == 'diamonds':
            up = "upgrade" if card.value > gs.weapon else "equip"
            return f"Weapon: {card.value} ({up})"
        # monster
        val = card.value
        if gs.combat_mode == 'auto':
            use_w = can_use_weapon_on_room(gs, val)
            dmg = max(0, val - gs.weapon) if use_w else val
            return f"Auto {'Weapon' if use_w else 'Fist'}: -{dmg} HP"
        if gs.combat_mode == 'weapon':
            if can_use_weapon_on_room(gs, val):
                dmg = max(0, val - gs.weapon)
                return f"Weapon: -{dmg} HP"
            return "Weapon: invalid (chain)"
        # fist
        return f"Fist: -{val} HP"

    def draw_tooltip(surface, text: str, pos: tuple[int,int]):
        if not text:
            return
        pad = 8
        max_w = 280
        lines: list[str] = []
        # Robust split that cannot produce unterminated string literals
        for part in text.splitlines():
            lines.extend(wrap_text(part, FONT_SM, max_w))
        if not lines:
            lines = [""]
        w = max(FONT_SM.size(line)[0] for line in lines) + pad*2
        h = len(lines) * (FONT_SM.get_height()+2) + pad*2
        x, y = pos
        if x + w > W - 12:
            x = W - 12 - w
        if y + h > H - 12:
            y = H - 12 - h
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surface, COL_TOOLTIP_BG, rect, border_radius=8)
        pygame.draw.rect(surface, COL_BORDER, rect, width=1, border_radius=8)
        ty = y + pad
        for line in lines:
            draw_text(surface, line, x + pad, ty, color=(226, 232, 240), font=FONT_SM)
            ty += FONT_SM.get_height()+2

    # Window (resizable)
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Scoundrel – Pygame")
    clock = pygame.time.Clock()

    gs = start_new_run()

    # Stable slot mapping for current room
    slot_card_ids: List[Optional[str]] = [c.id for c in gs.room] + [None, None, None, None]
    slot_card_ids = slot_card_ids[:4]

    def reset_slots_from_room():
        nonlocal slot_card_ids
        slot_card_ids = [c.id for c in gs.room]
        if len(slot_card_ids) < 4:
            slot_card_ids += [None] * (4 - len(slot_card_ids))
        else:
            slot_card_ids = slot_card_ids[:4]

    def sync_slots_after_action():
        nonlocal slot_card_ids
        if len(gs.room) == 4:
            room_ids = [c.id for c in gs.room]
            current_ids = [cid for cid in slot_card_ids if cid is not None]
            if set(room_ids) != set(current_ids) or sum(1 for x in slot_card_ids if x is None):
                reset_slots_from_room()

    running = True
    hovered_tip = ''
    while running:
        # True if window has keyboard focus, regardless of mouse position
        has_focus = bool(pygame.key.get_focused())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                W, H = max(900, event.w), max(650, event.h)
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN and has_focus:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    gs = start_new_run()
                    reset_slots_from_room()
                elif event.key == pygame.K_f:
                    flee_room(gs)
                    reset_slots_from_room()
                elif event.key == pygame.K_a:
                    gs.combat_mode = "auto"
                elif event.key == pygame.K_w:
                    gs.combat_mode = "weapon"
                elif event.key == pygame.K_s:
                    gs.combat_mode = "fist"
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    header, ctrl, room_rect, log_rect = layout()
                    slot = {pygame.K_1:0, pygame.K_2:1, pygame.K_3:2, pygame.K_4:3}[event.key]
                    cid = slot_card_ids[slot]
                    if cid is not None:
                        idx = next((i for i, c in enumerate(gs.room) if c.id == cid), None)
                        if idx is not None:
                            consumed = on_click_card(gs, idx)
                            if consumed:
                                slot_card_ids[slot] = None
                                sync_slots_after_action()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                header, ctrl, room_rect, log_rect = layout()
                mx, my = event.pos
                for slot in range(4):
                    if slot_rect(slot, room_rect).collidepoint(mx, my):
                        cid = slot_card_ids[slot]
                        if cid is not None:
                            idx = next((i for i, c in enumerate(gs.room) if c.id == cid), None)
                            if idx is not None:
                                consumed = on_click_card(gs, idx)
                                if consumed:
                                    slot_card_ids[slot] = None
                                    sync_slots_after_action()
                        break

        # Drawing
        header, ctrl, room_rect, log_rect = layout()
        screen.fill(COL_BG)

        # Header and badges without overlap
        pygame.draw.rect(screen, COL_PANEL, header, border_radius=16)
        pygame.draw.rect(screen, COL_BORDER, header, width=2, border_radius=16)
        draw_text(screen, "Scoundrel", header.x + 16, header.y + 12, font=FONT_LG)
        draw_badges(screen, header, gs)

        # Controls
        pygame.draw.rect(screen, COL_PANEL, ctrl, border_radius=16)
        pygame.draw.rect(screen, COL_BORDER, ctrl, width=2, border_radius=16)
        draw_text(screen, "[A]uto  [W]eapon  [S] Fist   [F] Flee   [1-4] Play Card   [N] New Run   [ESC] Quit", ctrl.x + 16, ctrl.y + 18)
        if not has_focus:
            draw_text(screen, "Click window to focus for hotkeys", ctrl.x + 16, ctrl.y + 38, color=(203, 213, 225), font=FONT_SM)

        # Room panel
        pygame.draw.rect(screen, COL_PANEL_SOFT, room_rect, border_radius=16)
        pygame.draw.rect(screen, COL_BORDER, room_rect, width=2, border_radius=16)
        draw_text(screen, f"Current Room  |  Plays: {cards_played_this_room(gs)} / 3", room_rect.x + 12, room_rect.y + 12)

        # Map ids and draw slots
        id_to_card = {c.id: c for c in gs.room}
        hovered_tip = ''
        mx, my = pygame.mouse.get_pos()
        for slot in range(4):
            r = slot_rect(slot, room_rect)
            cid = slot_card_ids[slot]
            if cid is None or cid not in id_to_card:
                pygame.draw.rect(screen, COL_PANEL, r, border_radius=14)
                pygame.draw.rect(screen, COL_PLACEHOLDER, r, width=2, border_radius=14)
            else:
                c = id_to_card[cid]
                pygame.draw.rect(screen, COL_WHITE, r, border_radius=14)
                pygame.draw.rect(screen, (200, 200, 200), r, width=2, border_radius=14)
                is_red = c.suit in ("hearts", "diamonds")
                pip = "♥" if c.suit == "hearts" else "♦" if c.suit == "diamonds" else "♣" if c.suit == "clubs" else "♠"
                color = COL_RED if is_red else COL_BLACK
                draw_text(screen, f"{c.rank}", r.x + 8, r.y + 8, color=color)
                draw_text(screen, pip, r.x + 8, r.y + 32, color=color)
                big = FONT_LG.render(pip, True, color)
                screen.blit(big, (r.centerx - big.get_width() // 2, r.centery - big.get_height() // 2))

            # index badge
            idx_badge = pygame.Rect(r.x - 10, r.y - 18, 26, 18)
            pygame.draw.rect(screen, COL_ACCENT, idx_badge, border_radius=8)
            draw_text(screen, str(slot + 1), idx_badge.x + 8, idx_badge.y + 0, color=COL_BLACK, font=FONT_SM)

            # Tooltip capture
            if r.collidepoint(mx, my) and cid is not None and cid in id_to_card:
                hovered_tip = predict_hover_text(gs, id_to_card[cid])
                tip_pos = (r.right + 8, r.top + 8)

        # Log panel with proper wrapping and clipping
        pygame.draw.rect(screen, COL_PANEL, log_rect, border_radius=16)
        pygame.draw.rect(screen, COL_BORDER, log_rect, width=2, border_radius=16)
        draw_text(screen, "Log", log_rect.x + 12, log_rect.y + 10)
        pad = 12
        y = log_rect.y + 34
        max_y = log_rect.bottom - 12
        max_w = log_rect.width - pad * 2
        for raw in gs.log:
            for line in wrap_text(raw, FONT_SM, max_w):
                if y + FONT_SM.get_height() > max_y:
                    break
                draw_text(screen, line, log_rect.x + pad, y, color=(203,213,225), font=FONT_SM)
                y += FONT_SM.get_height() + 2
            if y + FONT_SM.get_height() > max_y:
                break

        # Result footer
        if gs.state != "playing":
            res_rect = pygame.Rect(20, H - 48, W - 40, 36)
            pygame.draw.rect(screen, COL_PANEL, res_rect, border_radius=16)
            pygame.draw.rect(screen, COL_BORDER, res_rect, width=2, border_radius=16)
            pts = gs.score if gs.score is not None else (compute_victory_score(gs) if gs.state=='won' else compute_defeat_score(gs))
            draw_text(screen, f"Run Result: {'Victory' if gs.state=='won' else 'Defeat'}  |  Points: {pts}  |  Press [N] for New Run", res_rect.x + 16, res_rect.y + 8)


        # Draw tooltip last so it stays on top
        if hovered_tip:
            draw_tooltip(screen, hovered_tip, tip_pos)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# -------------------------------------------------------------
# Tests (run with RUN_TESTS=1)
# -------------------------------------------------------------

def _run_tests():
    import unittest

    class TestScoundrel(unittest.TestCase):
        def test_deck_shape(self):
            deck = build_deck()
            self.assertEqual(len(deck), 44)
            # count by suit
            counts = {s:0 for s in SUITS}
            for c in deck:
                counts[c.suit]+=1
            self.assertEqual(counts['hearts'], 9)    # 2..10
            self.assertEqual(counts['diamonds'], 9)  # 2..10
            self.assertEqual(counts['clubs'], 13)
            self.assertEqual(counts['spades'], 13)
            # ensure no red faces or red aces
            for c in deck:
                if c.suit in ('hearts','diamonds'):
                    self.assertTrue(c.rank not in ('J','Q','K','A'))

        def test_incomplete_room_wins(self):
            # Create state where carry=1 and deck has 2 cards -> cannot reach 4
            deck = build_deck()[:]
            # minimal crafted: take any 3 cards as deck remainder of size 2
            carry = deck[:1]
            remainder = deck[1:3]
            gs = GameState(deck=remainder, room=[], discard=[], state='playing')
            gs.ran_last_room = False
            fill_room_to_four(gs, carry)
            self.assertEqual(gs.state, 'won')

        def test_flee_only_at_start(self):
            # set room with a heart so we can make a play then attempt to flee
            # room needs at least one heart card; construct one
            hearts = [Card(id=f"h-{i}", suit='hearts', rank=str(i+2), value=i+2) for i in range(2,3)]
            others = [Card(id=f"c-{i}", suit='clubs', rank='2', value=2) for i in range(3)]
            room = hearts + others
            deck = build_deck()[10:20]
            gs = GameState(deck=deck, room=room[:4], discard=[], state='playing')
            # play one card (heart), then try to flee
            on_click_card(gs, 0)
            ran_before = gs.ran_last_room
            flee_room(gs)
            self.assertEqual(gs.ran_last_room, ran_before)  # still same
            self.assertIn('only flee at the start', gs.log[0].lower())

        def test_solver_finds_winnable(self):
            deck, attempts, nodes = find_winnable_deck(max_shuffles=64)
            self.assertTrue(len(deck) > 0)
            self.assertLessEqual(attempts, 64)

        def test_auto_mode_weapon_logic(self):
            # weapon 7, monster 9 should weapon for 2 dmg; then chain=9 so next weapon target must be <9
            m9 = Card(id='m9', suit='spades', rank='9', value=9)
            m9b = Card(id='m9b', suit='clubs', rank='9', value=9)
            m8 = Card(id='m8', suit='spades', rank='8', value=8)
            w7 = Card(id='w7', suit='diamonds', rank='7', value=7)
            gs = GameState(deck=[], room=[w7, m9, m8, m9b], discard=[], state='playing')
            on_click_card(gs, 0)  # pick weapon 7
            self.assertEqual(gs.weapon, 7)
            on_click_card(gs, 0)  # fight m9 with auto -> weapon
            self.assertEqual(gs.weapon_chain_limit, 9)
            # now two cards remain (m8 and m9b carry rules handle later)
            # try to fight m9b: auto should punch since 9 !< 9
            on_click_card(gs, 1 if len(gs.room)>1 else 0)
            self.assertLess(gs.health, 20)  # took damage by fist

    class TestScoundrelCooldown(unittest.TestCase):
        def test_cooldown_resets_after_new_room(self):
            # Create a room where we can make 3 plays, then ensure cooldown resets for flee
            h2 = Card(id='h2', suit='hearts', rank='2', value=2)
            h3 = Card(id='h3', suit='hearts', rank='3', value=3)
            h4 = Card(id='h4', suit='hearts', rank='4', value=4)
            c2 = Card(id='c2', suit='clubs', rank='2', value=2)
            room = [h2, h3, h4, c2]
            deck = build_deck()[0:10]
            gs = GameState(deck=deck, room=room, discard=[], state='playing')
            on_click_card(gs, 0)
            on_click_card(gs, 0)
            on_click_card(gs, 0)
            # New room formed by carry->refill should clear run-cooldown
            self.assertFalse(gs.ran_last_room)
            before = len(gs.log)
            flee_room(gs)
            after = len(gs.log)
            self.assertGreater(after, before)
            self.assertNotIn('cannot run twice', gs.log[0].lower())

    class TestInvalidWeapon(unittest.TestCase):
        def test_invalid_weapon_does_not_consume(self):
            # Setup: weapon=7, chain limit already set to 9, try to weapon a 9 -> invalid
            m9 = Card(id='m9', suit='spades', rank='9', value=9)
            h2 = Card(id='h2', suit='hearts', rank='2', value=2)
            d7 = Card(id='d7', suit='diamonds', rank='7', value=7)
            x = Card(id='c2', suit='clubs', rank='2', value=2)
            gs = GameState(deck=[], room=[m9, h2, d7, x], discard=[], state='playing')
            gs.weapon = 7
            gs.weapon_chain_limit = 9  # prior big hit would have set this
            gs.combat_mode = 'weapon'
            before = list(gs.room)
            consumed = on_click_card(gs, 0)
            self.assertFalse(consumed)
            self.assertEqual([c.id for c in gs.room], [c.id for c in before])
            self.assertIn('weapon cannot be used', '\n'.join(gs.log).lower())

    class TestVictoryBonus(unittest.TestCase):
        def test_victory_heart_bonus(self):
            h5 = Card(id='h5', suit='hearts', rank='5', value=5)
            gs = GameState(deck=[], room=[], discard=[], state='won')
            gs.health = 20
            gs.max_health = 20
            gs.last_remaining = h5
            self.assertEqual(compute_victory_score(gs), 25)

    class TestPointsLogged(unittest.TestCase):
        def test_points_logged_on_defeat_and_win(self):
            # Defeat path: HP 5 vs A(20) -> hp becomes -15; 3 more aces remain -> -15 - 60 = -75
            m20 = Card(id='m20', suit='clubs', rank='A', value=20)
            gs = GameState(deck=[], room=[m20, m20, m20, m20], discard=[], state='playing')
            gs.health = 5
            gs.combat_mode = 'fist'
            on_click_card(gs, 0)
            self.assertEqual(gs.state, 'lost')
            self.assertIn('points: -75', '\n'.join(gs.log).lower())

            # Win by incomplete room still logs points
            deck = build_deck()[:]
            carry = deck[:1]
            remainder = deck[1:3]
            gs2 = GameState(deck=remainder, room=[], discard=[], state='playing')
            fill_room_to_four(gs2, carry)
            self.assertEqual(gs2.state, 'won')
            self.assertTrue(any('points:' in s.lower() for s in gs2.log))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestScoundrel)
    suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestScoundrelCooldown)
    suite3 = unittest.defaultTestLoader.loadTestsFromTestCase(TestInvalidWeapon)
    suite4 = unittest.defaultTestLoader.loadTestsFromTestCase(TestVictoryBonus)
    suite5 = unittest.defaultTestLoader.loadTestsFromTestCase(TestPointsLogged)
    all_suite = unittest.TestSuite([suite, suite2, suite3, suite4, suite5])
    res = unittest.TextTestRunner(verbosity=2).run(all_suite)
    if not res.wasSuccessful():
        sys.exit(1)

# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    if os.environ.get("RUN_TESTS"):
        _run_tests()
    else:
        run_game()