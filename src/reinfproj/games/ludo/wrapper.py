from typing import Literal
import cv2
import ludopy

from reinfproj.games.ludo.constants import LUDO_SAFE_POS_SET
from reinfproj.games.ludo.ludo_obs import LudoObs, LudoPyBaseObs


RenderMode = Literal["human", "rgb_array"] | None


class Ludo:
    game: ludopy.Game
    render_mode: RenderMode

    __win_title: str | None

    def __init__(
        self, ghost_players: list[int] | None = None, *, render_mode: RenderMode = None
    ):
        self.game = ludopy.Game(ghost_players=ghost_players)
        self.render_mode = render_mode
        if render_mode == "human":
            self.__win_title = "Ludo"
        else:
            self.__win_title = None

    @staticmethod
    def game_obs_to_state(base_obs: LudoPyBaseObs, player_idx: int) -> LudoObs:
        return LudoObs.from_base(base_obs, player_idx)

    def reset(self):
        self.game.reset()

    def observe(self):
        ludo_base_obs: LudoPyBaseObs
        ludo_base_obs, player_idx = self.game.get_observation()
        # print(ludo_base_obs, player_idx)
        obs = LudoObs.from_base(ludo_base_obs, player_idx)

        # if player_idx == 0:
        #     return obs, obs
        #
        # p_pieces: PiecesArr
        # e_pieces: EnemyPiecesArr
        # p_pieces, e_pieces = self.game.get_pieces(seen_from=player_idx)

        # subjective_obs = LudoObs.subjective_obs(obs, p_pieces, e_pieces)

        return obs

    def step(self, action: int, previous_obs: LudoObs):
        player_idx = self.game.current_player

        ludo_base_obs: LudoPyBaseObs
        ludo_base_obs = self.game.answer_observation(action)

        new_obs = LudoObs.from_base(ludo_base_obs, player_idx)

        if action == -1:
            return new_obs, 0.0

        reward = self.calc_reward(previous_obs, new_obs, action)

        return new_obs, reward

    def render(self):
        if self.render_mode is None:
            return None

        env_rgb = self.game.render_environment()
        if self.render_mode == "rgb_array":
            return env_rgb

        win_title = self.__win_title or "Ludo"

        env_bgr = cv2.cvtColor(env_rgb, cv2.COLOR_RGB2BGR)
        _ = cv2.imshow(win_title, env_bgr)
        _ = cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def save_to_file(self, file_path: str):
        self.game.save_hist(file_path)

    def calc_reward(self, old_obs: LudoObs, new_obs: LudoObs, action: int) -> float:
        if action == -1:
            return 0.0

        return self.calc_reward_old(old_obs, new_obs, action)

    def calc_reward_old(self, old_obs: LudoObs, new_obs: LudoObs, action: int) -> float:
        # Rewards table for reference:
        # Move out  -> 0.25
        # Normal    -> 0.01
        # Goal      -> 0.8
        # Star      -> 0.5
        # Globe     -> 0.4
        # Protect   -> 0.3
        # Kill      -> 0.4
        # Die       -> -0.5
        # Goal Zone -> 0.4
        # Nothing   -> 0.0

        old_pos: int = old_obs.player_pieces[action].item()
        new_pos: int = new_obs.player_pieces[action].item()

        # 1) If the piece didn't move at all => "Nothing" (0.0)
        if old_pos == new_pos:
            return 0.0

        # 2) Check if we "died":
        #    If we were on board before but are now back in home (e.g. new_pos < 0)
        #    That means we got kicked out.
        if old_pos > 0 and new_pos == 0:
            return -0.5  # "Die"

        # 3) "Move out": if we were in home (< 0) and moved onto the board (>= 0)
        if old_pos == 0 and new_pos > 0:
            return 0.25

        # 4) "Kill": if we caused an enemy to go back to home.
        #    Simplest check: if any enemy was on new_pos in old_obs, but is not there in new_obs.
        #    (And new_pos is not safe for the enemy.)
        #    We'll do a short helper function here:
        def count_enemies_in(pos: int, enemy_pieces: list[list[int]]):
            c = 0
            for ep in enemy_pieces:
                for epos in ep:
                    if int(epos) == pos:
                        c += 1
            return c

        old_enemies_on_new = count_enemies_in(new_pos, old_obs.enemy_pieces)
        new_enemies_on_new = count_enemies_in(new_pos, new_obs.enemy_pieces)

        #    If at least one enemy was there and is now gone => we killed them
        #    (assuming not a safe square that protects them).
        #    Typically, a globe protects everyone, so you can't kill on a globe.
        #    We'll do a quick check that new_pos isn't a globe (and not a multi-same color stack).
        if old_enemies_on_new > 0 and new_enemies_on_new < old_enemies_on_new:
            # For a real environment, you'd likely also check if new_pos is safe for the enemy.
            # But to keep things simple we just say we got a kill.
            return 0.4

        # 5) "Goal": if piece has reached the final “goal” position.  Usually that’s position 57 or 58, or 59 if your internal representation goes that high.
        #    Check new_pos == 58 (common in Ludopy is that 58 is the final)
        if new_pos == 58:
            return 0.8

        # 6) "Goal Zone": if the piece is in final stretch (like positions 52..57).
        #    Adjust if your board indexing differs.
        if 52 <= new_pos <= 57:
            return 0.4

        # 7) "Star": if new_pos is in star_positions
        # if new_pos in star_positions:
        #     return 0.5

        # 8) "Globe": if new_pos is in globe_positions
        if new_pos in LUDO_SAFE_POS_SET:
            return 0.4

        # 9) "Protect": if we land on a position that’s occupied by another one of our own pieces
        #    -> multiple friendly pieces stack on same position => cannot be killed.
        #    Count how many of our own pieces (other than the one that moved) are at new_pos in new_obs
        protect_count = sum(
            (
                1
                for i, p in enumerate(new_obs.player_pieces)
                if i != action and p == new_pos
            )
        )
        if protect_count > 0:
            return 0.3

        # 10) "Normal": everything else => 0.01
        return 0.01
