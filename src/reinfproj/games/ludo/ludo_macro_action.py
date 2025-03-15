from enum import IntEnum
from reinfproj.games.ludo.constants import LUDO_SAFE_POS_SET, LUDO_STAR_POS_SET
from reinfproj.games.ludo.ludo_obs import LudoObs


class MacroAction(IntEnum):
    MOVE_OUT = 0
    NORMAL = 1
    GOAL = 2
    STAR = 3
    GLOBE = 4
    PROTECT = 5
    KILL = 6
    DIE = 7
    GOAL_ZONE = 8


# Define some board-specific sets
# star_positions = {5, 11, 18, 24, 31, 37, 44, 50}


def classify_macro_action_for_piece(obs: LudoObs, piece_idx: int) -> MacroAction:
    """
    Given the current observation and a piece index, classify what macro action
    would result if we moved that piece by obs.dice steps.
    (Assumes piece_idx is in obs.move_pieces.)
    """
    old_pos = int(obs.player_pieces[piece_idx])
    dice = obs.dice
    new_pos = old_pos + dice

    # Move-out if starting from home with a 6
    if old_pos == 0 and dice == 6:
        return MacroAction.MOVE_OUT
    # If already on board and we get bumped back to home, mark as DIE
    if old_pos > 0 and new_pos == 0:
        return MacroAction.DIE
    # Moving out from home
    if old_pos == 0 and new_pos > 0:
        return MacroAction.MOVE_OUT
    # If piece exactly reaches the goal (assume final is 58)
    if new_pos == 58:
        return MacroAction.GOAL
    # If piece is in the goal zone (positions 52 to 57)
    if 52 <= new_pos <= 57:
        return MacroAction.GOAL_ZONE
    # If an enemy is on new_pos, assume we kill them (simplified)
    if enemy_on_position(obs, new_pos):
        return MacroAction.KILL
    # If landing on a star
    if new_pos in LUDO_STAR_POS_SET:
        return MacroAction.STAR
    # If landing on a globe
    if new_pos in LUDO_SAFE_POS_SET:
        return MacroAction.GLOBE
    # If stacking on our own piece
    if stacked_on_own(obs, piece_idx, new_pos):
        return MacroAction.PROTECT
    # Otherwise, a normal move
    return MacroAction.NORMAL


def enemy_on_position(obs: LudoObs, pos: int) -> bool:
    """
    Returns True if any enemy piece occupies 'pos'.
    """
    for enemy_arr in obs.enemy_pieces:
        for epos in enemy_arr:
            if int(epos) == pos:
                return True
    return False


def stacked_on_own(obs: LudoObs, piece_idx: int, new_pos: int) -> bool:
    """
    Returns True if moving piece_idx to new_pos would result in stacking on another of our pieces.
    """
    for i, p in enumerate(obs.player_pieces):
        if i == piece_idx:
            continue
        if int(p) == new_pos:
            return True
    return False


def get_feasible_macro_actions(obs: LudoObs) -> list[MacroAction]:
    """
    Returns a list of macro actions (as MacroAction values) that are feasible
    given the current move pieces.
    """
    feasible: set[MacroAction] = set()
    for piece_idx in obs.move_pieces:
        act = classify_macro_action_for_piece(obs, piece_idx)
        if act is not None:
            feasible.add(act)
    return list(feasible)


def choose_piece_for_macro(obs: LudoObs, macro_action: MacroAction) -> int:
    """
    Given a desired macro action, return the first piece index in obs.move_pieces
    whose classification matches the macro action. Return -1 if none.
    """
    for piece_idx in obs.move_pieces:
        act = classify_macro_action_for_piece(obs, piece_idx)
        if act == macro_action:
            return piece_idx
    return -1
