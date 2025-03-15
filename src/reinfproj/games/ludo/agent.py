from typing import Protocol
from typing_extensions import override
from reinfproj.games.ludo.ludo_obs import LudoObs
from reinfproj.games.ludo.strats import LudoStrategy


class LudoAgent(Protocol):
    def get_action(self, obs: LudoObs) -> int: ...


class RandomAgent(LudoAgent):
    def __init__(self):
        self.num_moves: int = 0

    @override
    def get_action(self, obs: LudoObs) -> int:
        if obs.move_pieces.size == 0:
            return -1

        self.num_moves += 1
        if obs.move_pieces.size == 1:
            return int(obs.move_pieces[0].item())

        move = LudoStrategy.random_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )

        return move


class FastAgent(LudoAgent):
    @override
    def get_action(self, obs: LudoObs) -> int:
        if obs.move_pieces.size == 0:
            return -1

        if obs.move_pieces.size == 1:
            return int(obs.move_pieces[0].item())

        move = LudoStrategy.fast_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )

        if move is not None:
            return move
        else:
            return LudoStrategy.random_move(
                obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
            )


class AggressiveAgent(LudoAgent):
    @override
    def get_action(self, obs: LudoObs) -> int:
        if obs.move_pieces.size == 0:
            return -1

        if obs.move_pieces.size == 1:
            return int(obs.move_pieces[0].item())

        move = LudoStrategy.aggressive_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )

        if move is not None:
            return move
        else:
            return LudoStrategy.random_move(
                obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
            )


class DefensiveAgent(LudoAgent):
    def __init__(self):
        self.num_moves: int = 0
        self.used_random: int = 0

    @override
    def get_action(self, obs: LudoObs) -> int:
        if obs.move_pieces.size == 0:
            return -1

        if obs.move_pieces.size == 1:
            return int(obs.move_pieces[0].item())

        self.num_moves += 1

        move = LudoStrategy.defensive_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )

        if move is not None:
            return move
        else:
            self.used_random += 1
            return LudoStrategy.random_move(
                obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
            )


class ExpertAgent(LudoAgent):
    @override
    def get_action(self, obs: LudoObs) -> int:
        if obs.move_pieces.size == 0:
            return -1

        if obs.move_pieces.size == 1:
            return int(obs.move_pieces[0].item())

        # print("checking defensive move")
        move = LudoStrategy.defensive_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )
        if move is not None:
            return move

        # print("checking aggressive move")
        move = LudoStrategy.aggressive_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )
        if move is not None:
            return move

        # print("checking fast move")
        move = LudoStrategy.fast_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )
        if move is not None:
            return move

        # print("checking random move")

        return LudoStrategy.random_move(
            obs.dice, obs.move_pieces, obs.player_pieces, obs.enemy_pieces
        )
