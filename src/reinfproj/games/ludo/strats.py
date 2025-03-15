from reinfproj.games.ludo.ludo_obs import (
    PiecesArr,
    EnemyPiecesArr,
    MovePiecesArr,
)
import numpy as np
from reinfproj.games.ludo.constants import LUDO_SAFE_POS_SET


class LudoStrategy:
    num_random_calls: int = 0

    # A defensive move: moves player piece out of knocking range (1-6)
    @staticmethod
    def defensive_move(
        _dice: int,
        move_pieces: MovePiecesArr,
        player_pieces: PiecesArr,
        enemy_pieces: EnemyPiecesArr,
    ) -> int | None:
        if move_pieces.size == 0:
            return None

        for piece_o in move_pieces:
            piece: int = piece_o.item()
            pos: int = player_pieces[piece].item()
            if pos == 0:
                continue
            knockout_range = range(pos - 5, pos + 1)

            for enemy_num in enemy_pieces:
                for enemy in enemy_num:
                    ep: int = enemy.item()
                    if ep == 0:  # sitting at home
                        continue
                    if ep in knockout_range:
                        return piece

        return None

    # knock out enemy piece if possible
    @staticmethod
    def aggressive_move(
        dice: int,
        move_pieces: MovePiecesArr,
        player_pieces: PiecesArr,
        enemy_pieces: EnemyPiecesArr,
    ) -> int | None:
        if move_pieces.size == 0:
            return None

        for piece_o in move_pieces:
            piece: int = piece_o.item()
            new_player_pos: int = player_pieces[piece].item() + dice

            for enemy_num in enemy_pieces:
                for enemy in enemy_num:
                    ep: int = enemy.item()
                    if ep == 0 or ep in LUDO_SAFE_POS_SET:
                        continue

                    if ep == new_player_pos:
                        return piece

        return None

    #
    @staticmethod
    def fast_move(
        _dice: int,
        move_pieces: MovePiecesArr,
        player_pieces: PiecesArr,
        _enemy_pieces: EnemyPiecesArr,
    ) -> int | None:
        if move_pieces.size == 0:
            return None

        return max(move_pieces, key=lambda p: player_pieces[p.item()]).item()

    @staticmethod
    def random_move(
        _dice: int,
        move_pieces: MovePiecesArr,
        _player_pieces: PiecesArr,
        _enemy_pieces: EnemyPiecesArr,
    ) -> int:
        if move_pieces.size == 0:
            return -1

        LudoStrategy.num_random_calls += 1
        return move_pieces[np.random.randint(len(move_pieces))].item()
