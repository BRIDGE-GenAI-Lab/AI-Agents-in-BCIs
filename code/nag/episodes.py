"""Deterministic episode construction from real bigP3BCI online decodes."""
from __future__ import annotations
import logging
import pandas as pd

ALS_STUDIES = ("StudyB", "StudyF", "StudyL", "StudyN")

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_123456789"

logger = logging.getLogger(__name__)


def _to_char(code) -> str | None:
    """Grid code -> single character, 1-indexed over the 6x6 matrix, or None
    if the code is out of range.

    bigP3BCI uses different speller matrix sizes across studies: across the
    full dataset `target` reaches 54 and `selected` reaches 72. The 6x6
    alphabet below is correct only for the 6x6 grid (verified against real
    EDF channel names: A_1_1 .. Z_5_2, Sp_5_3, 1_5_4 .. 9_6_6, where "_" is
    the Sp/space cell at index 27). A code outside 1..36 must never be
    silently mapped to a placeholder character -- a corrupted string later
    scored as a transmission failure would be a fabricated result.
    """
    try:
        i = int(code) - 1
    except (TypeError, ValueError):
        return None
    return _ALPHABET[i] if 0 <= i < len(_ALPHABET) else None


def build_episodes(df: pd.DataFrame, length: int = 5, als_only: bool = True) -> pd.DataFrame:
    """Chunk consecutive eligible selections into fixed-length episodes.

    Episodes never span files (a file is one recording block), so a decoded
    string is always a contiguous real transmission. Trailing partial chunks
    are dropped so every episode has identical length. Any episode
    containing a grid code outside 1..36 (see `_to_char`) is dropped
    entirely -- never rendered with a placeholder character -- and the
    total dropped is logged so the drop is observable, not silent.
    """
    d = df[df["eligible"] == True].copy()  # noqa: E712 - explicit for object dtype
    d = d.dropna(subset=["target", "selected"])
    if als_only:
        d = d[d["study"].isin(ALS_STUDIES)]
    d = d.sort_values(["relative_path", "trial_number"])

    out = []
    n_dropped_episodes = 0
    n_dropped_codes = 0
    for path, g in d.groupby("relative_path", sort=False):
        n_full = len(g) // length
        for k in range(n_full):
            chunk = g.iloc[k * length:(k + 1) * length]
            true_chars = [_to_char(c) for c in chunk["target"]]
            dec_chars = [_to_char(c) for c in chunk["selected"]]
            n_out_of_range = sum(c is None for c in true_chars) + sum(c is None for c in dec_chars)
            if n_out_of_range > 0:
                n_dropped_episodes += 1
                n_dropped_codes += n_out_of_range
                continue
            true_s = "".join(true_chars)
            dec_s = "".join(dec_chars)
            t = chunk["phase3_time_seconds"].to_numpy(dtype=float)
            out.append(dict(
                episode_id=f"{path}#{k:04d}",
                participant_id=chunk["study_participant_id"].iloc[0],
                session_id=chunk["session_id"].iloc[0],
                study=chunk["study"].iloc[0],
                condition=chunk["condition"].iloc[0],
                true_string=true_s,
                decoded_string=dec_s,
                n_selections=length,
                n_errors=int(sum(a != b for a, b in zip(true_s, dec_s))),
                n_out_of_range=0,
                duration_s=float(t.max() - t.min()) if len(t) > 1 else 0.0,
                stratum="ALS" if chunk["study"].iloc[0] in ALS_STUDIES else "able-bodied",
            ))
    if n_dropped_episodes:
        logger.warning(
            "build_episodes: dropped %d episode(s) containing %d total out-of-range "
            "grid code(s) (codes outside 1..36) rather than rendering them as a "
            "placeholder character",
            n_dropped_episodes, n_dropped_codes,
        )
    return pd.DataFrame(out, columns=[
        "episode_id", "participant_id", "session_id", "study", "condition",
        "true_string", "decoded_string", "n_selections", "n_errors",
        "n_out_of_range", "duration_s", "stratum"])
