from typing import Self, TypeAlias
import numpy as np
from dataclasses import dataclass

from reinfproj.games.ludo.constants import LUDO_SAFE_POS_SET


StateArr: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FlatStateArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float32]]
MovePiecesArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
PiecesArr: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint16]]
EnemyPiecesArr: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint16]]

LudoPyBaseObs = tuple[int, MovePiecesArr, PiecesArr, EnemyPiecesArr, bool, bool]
LudoPyObs = tuple[LudoPyBaseObs, int]


def classify_macro(pos: int) -> int:
    if pos == 0:
        return 0  # Home

    if pos >= 53 or pos in LUDO_SAFE_POS_SET:
        return 1  # SAFE

    return 2  # UNSAFE


def update_state(
    idx: int,
    state: StateArr,
    pieces: PiecesArr,
):
    for v in pieces:
        state[idx, v] += 1


@dataclass
class LudoObs:
    dice: int
    move_pieces: MovePiecesArr
    player_pieces: PiecesArr
    enemy_pieces: EnemyPiecesArr
    player_idx: int
    there_is_a_winner: bool
    player_is_a_winner: bool

    state: StateArr

    @classmethod
    def from_base(cls, base_obs: LudoPyBaseObs, player_idx: int):
        state = np.zeros((4, 59), dtype=np.float32)

        (
            dice,
            move_pieces,
            player_pieces,
            enemy_pieces,
            player_is_winner,
            there_is_winner,
        ) = base_obs

        update_state(0, state, player_pieces)

        for i in range(3):
            update_state(i + 1, state, enemy_pieces[i])

        return cls(
            dice=dice,
            move_pieces=move_pieces,
            player_pieces=player_pieces,
            enemy_pieces=enemy_pieces,
            player_idx=player_idx,
            there_is_a_winner=there_is_winner,
            player_is_a_winner=player_is_winner,
            state=state,
        )

    @classmethod
    def subjective_obs(
        cls, normal: Self, p_pieces: PiecesArr, e_pieces: EnemyPiecesArr
    ) -> Self:
        state = np.zeros((4, 59), dtype=np.float32)

        update_state(0, state, p_pieces)
        for i in range(3):
            update_state(i + 1, state, e_pieces[i])

        return cls(
            dice=normal.dice,
            move_pieces=normal.move_pieces,
            player_pieces=p_pieces,
            enemy_pieces=e_pieces,
            player_idx=normal.player_idx,
            there_is_a_winner=normal.there_is_a_winner,
            player_is_a_winner=normal.player_is_a_winner,
            state=normal.state,
        )

    def is_over(self):
        return self.there_is_a_winner or self.player_is_a_winner

    def did_player_win(self):
        return self.player_is_a_winner

    def did_enemy_win(self):
        return self.there_is_a_winner and not self.player_is_a_winner

    def encode_micro(self) -> FlatStateArr:
        board = self.state.flatten()  # shape is (236,)

        dice_ohe = np.zeros(6, dtype=np.float32)
        dice_ohe[self.dice - 1] = 1.0

        can_move = np.zeros(4, dtype=np.float32)
        for mp in self.move_pieces:
            can_move[mp] = 1.0

        return np.concatenate([board, dice_ohe, can_move], axis=0)

    def encode_macro_state(self) -> FlatStateArr:
        # macro state, each piece can be home, safe or unsafe

        macro = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            pos = int(self.player_pieces[i])
            macro[0, i] = classify_macro(pos)  # 0=HOME,1=SAFE,2=UNSAFE

        for e in range(3):
            for j in range(4):
                pos = int(self.enemy_pieces[e, j])
                macro[e + 1, j] = classify_macro(pos)

        return macro.flatten()  # 16,
