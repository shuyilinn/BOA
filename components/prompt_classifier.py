"""
Prompt clustering for dynamic budget allocation (Optimization #5).

Classifies prompts into categories based on keyword matching:
  - bio_chem: synthesis, compounds, experiments → exploitable, needs deep search
  - gray: legal, strategy, social engineering → moderate difficulty
  - hard: hate speech, defamation, direct harm → hard reject, low budget

Budget allocation strategy:
  - bio_chem / gray → increase search depth and sample count
  - hard → early stop or low budget to save compute
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PromptCluster(Enum):
    BIO_CHEM = "bio_chem"
    GRAY = "gray"
    HARD = "hard"
    UNKNOWN = "unknown"


@dataclass
class ClusterResult:
    cluster: PromptCluster
    confidence: float  # 0..1
    matched_keywords: list[str]

    # Budget multipliers (applied to defaults)
    depth_multiplier: float = 1.0
    sample_multiplier: float = 1.0
    time_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

_BIO_CHEM_KEYWORDS = re.compile(
    r"(?i)\b("
    r"synthe[st]i[sz]e?|synthesis|compound|chemical|"
    r"experiment|laboratory|lab|reagent|precursor|"
    r"molecule|formula|reaction|catalyst|"
    r"biolog|pathogen|virus|bacteria|toxin|"
    r"explosive|detonat|ignit|combust|"
    r"drug|narcotic|methamphetamine|fentanyl|"
    r"weapon|firearm|ammunition|"
    r"nerve agent|poison|venom|lethal dose|"
    r"radioactive|uranium|plutonium|enrichment"
    r")\b"
)

_GRAY_KEYWORDS = re.compile(
    r"(?i)\b("
    r"bet|gambl|wager|odds|"
    r"law|legal|loophole|regulation|"
    r"strategy|tactic|scheme|plan|"
    r"persuad|manipulat|influence|social engineer|"
    r"bypass|circumvent|evade|avoid detection|"
    r"exploit|vulnerabilit|"
    r"phish|spoof|impersonat|"
    r"privacy|surveillance|track|monitor|"
    r"copyright|pirat|torrent|crack"
    r")\b"
)

_HARD_KEYWORDS = re.compile(
    r"(?i)\b("
    r"hate speech|hateful|racist|racial slur|"
    r"defam|slander|libel|"
    r"harass|bully|stalk|"
    r"child|minor|underage|"
    r"terroris|mass casualt|genocide|"
    r"suicide|self.harm|"
    r"graphic violence|torture|"
    r"sex traffic|human traffic"
    r")\b"
)


def classify_prompt(prompt: str) -> ClusterResult:
    """Classify a prompt into a cluster for budget allocation."""
    if not prompt:
        return ClusterResult(
            cluster=PromptCluster.UNKNOWN,
            confidence=0.0,
            matched_keywords=[],
        )

    bio_chem_matches = _BIO_CHEM_KEYWORDS.findall(prompt)
    gray_matches = _GRAY_KEYWORDS.findall(prompt)
    hard_matches = _HARD_KEYWORDS.findall(prompt)

    bio_score = len(bio_chem_matches)
    gray_score = len(gray_matches)
    hard_score = len(hard_matches)

    total = bio_score + gray_score + hard_score
    if total == 0:
        return ClusterResult(
            cluster=PromptCluster.UNKNOWN,
            confidence=0.0,
            matched_keywords=[],
            depth_multiplier=1.0,
            sample_multiplier=1.0,
            time_multiplier=1.0,
        )

    # Pick dominant cluster
    if hard_score >= bio_score and hard_score >= gray_score:
        return ClusterResult(
            cluster=PromptCluster.HARD,
            confidence=hard_score / total,
            matched_keywords=hard_matches,
            depth_multiplier=0.5,    # Less depth needed
            sample_multiplier=0.5,   # Fewer samples
            time_multiplier=0.5,     # Less time
        )

    if bio_score >= gray_score:
        return ClusterResult(
            cluster=PromptCluster.BIO_CHEM,
            confidence=bio_score / total,
            matched_keywords=bio_chem_matches,
            depth_multiplier=2.0,    # More depth
            sample_multiplier=1.5,   # More samples
            time_multiplier=2.0,     # More time
        )

    return ClusterResult(
        cluster=PromptCluster.GRAY,
        confidence=gray_score / total,
        matched_keywords=gray_matches,
        depth_multiplier=1.5,
        sample_multiplier=1.2,
        time_multiplier=1.5,
    )
