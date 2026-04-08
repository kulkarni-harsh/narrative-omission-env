"""Synthetic event and article factory for the Narrative Omission environment."""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ground truth data model
# ---------------------------------------------------------------------------

@dataclass
class EventGroundTruth:
    event_id: str
    event_type: str  # industrial_accident | political_decision | corporate_action | public_health
    who: str
    what: str
    when: str
    where: str
    why: str
    casualties: Optional[int]
    financial_impact: Optional[str]
    responsible_party: str
    cover_up_detail: str
    red_herring_source_id: Optional[str]
    tags: List[str]


@dataclass
class BiasProfile:
    source_name: str
    omitted_fields: List[str]
    framing: str  # neutral | pro_corporate | pro_government | sensationalist
    credibility_hint: bool


# ---------------------------------------------------------------------------
# Raw event templates (deterministic, no LLM)
# ---------------------------------------------------------------------------

_EVENT_TEMPLATES = [
    # ── industrial_accident (5) ──────────────────────────────────────────
    {
        "event_type": "industrial_accident",
        "who": "NovaChem Industries",
        "what": "A chlorine gas leak injured 34 workers and forced evacuation of nearby residents",
        "when": "March 12, 2024",
        "where": "Riverside Industrial Park, Ohio",
        "why": "Corroded valve seals that failed a safety inspection six months prior",
        "casualties": 34,
        "financial_impact": "$8.2 million in cleanup and medical costs",
        "responsible_party": "NovaChem Industries",
        "cover_up_detail": "Internal memos show management knew about the corroded valves for six months before the leak",
        "tags": ["environment", "corporate", "cover-up", "safety"],
    },
    {
        "event_type": "industrial_accident",
        "who": "Apex Mining Corp",
        "what": "A tunnel collapse trapped and killed 11 miners",
        "when": "January 5, 2024",
        "where": "Coalvale Mine, West Virginia",
        "why": "Ignored roof-support inspection reports filed by engineers",
        "casualties": 11,
        "financial_impact": "$15 million in liability",
        "responsible_party": "Apex Mining Corp",
        "cover_up_detail": "Three engineer reports warning of imminent collapse were deleted from company servers",
        "tags": ["mining", "corporate", "cover-up", "safety"],
    },
    {
        "event_type": "industrial_accident",
        "who": "PetroCrest LLC",
        "what": "An offshore oil spill released 400,000 gallons into the Gulf of Mexico",
        "when": "September 20, 2023",
        "where": "Gulf of Mexico, Block 47",
        "why": "Blowout preventer failure due to deferred maintenance",
        "casualties": None,
        "financial_impact": "$220 million estimated cleanup cost",
        "responsible_party": "PetroCrest LLC",
        "cover_up_detail": "PetroCrest submitted falsified maintenance logs to regulators claiming the blowout preventer was serviced 30 days prior",
        "tags": ["oil", "environment", "cover-up"],
    },
    {
        "event_type": "industrial_accident",
        "who": "TitanSteel Manufacturing",
        "what": "Toxic slag runoff contaminated the River Brent for 12 miles",
        "when": "July 8, 2024",
        "where": "Stanton, Pennsylvania",
        "why": "Improperly lined retention pond overflowed during heavy rain",
        "casualties": None,
        "financial_impact": "$45 million in remediation",
        "responsible_party": "TitanSteel Manufacturing",
        "cover_up_detail": "TitanSteel paid a local contractor to delay filing the EPA spill report by four days",
        "tags": ["water", "environment", "corporate"],
    },
    {
        "event_type": "industrial_accident",
        "who": "SolarWinds Fabrication",
        "what": "An explosion at a battery assembly plant killed 3 and injured 22",
        "when": "February 28, 2024",
        "where": "Phoenix, Arizona",
        "why": "Improper lithium storage in an area lacking explosion suppression systems",
        "casualties": 3,
        "financial_impact": "$30 million",
        "responsible_party": "SolarWinds Fabrication",
        "cover_up_detail": "The plant had been cited twice for the same storage violation and paid fines instead of installing suppression systems",
        "tags": ["manufacturing", "safety", "cover-up"],
    },
    # ── political_decision (5) ───────────────────────────────────────────
    {
        "event_type": "political_decision",
        "who": "Senator Dale Henshaw",
        "what": "Voted to roll back clean-air emission limits for coal plants",
        "when": "April 3, 2024",
        "where": "U.S. Senate, Washington D.C.",
        "why": "Framed as 'reducing energy costs for consumers'",
        "casualties": None,
        "financial_impact": "$2.1 billion in avoided compliance costs for coal industry",
        "responsible_party": "Senator Dale Henshaw",
        "cover_up_detail": "Senator Henshaw received $1.4 million in campaign contributions from coal industry PACs in the 12 months before the vote",
        "tags": ["politics", "environment", "corruption"],
    },
    {
        "event_type": "political_decision",
        "who": "Department of Agriculture",
        "what": "Cut food safety inspection frequency at large meat processing plants by 40%",
        "when": "October 15, 2023",
        "where": "Washington D.C.",
        "why": "Described as 'streamlining regulatory burden'",
        "casualties": None,
        "financial_impact": "$180 million saved by industry annually",
        "responsible_party": "Department of Agriculture",
        "cover_up_detail": "The decision followed two unannounced industry lobbying meetings that were never logged in public records",
        "tags": ["food safety", "regulatory rollback", "lobbying"],
    },
    {
        "event_type": "political_decision",
        "who": "Governor Maria Theron",
        "what": "Redirected $600 million in education funds to highway construction contracts",
        "when": "June 22, 2024",
        "where": "State Capitol, Texas",
        "why": "Cited job creation and infrastructure needs",
        "casualties": None,
        "financial_impact": "$600 million reallocation",
        "responsible_party": "Governor Maria Theron",
        "cover_up_detail": "Three highway contracts worth $280 million were awarded to firms that donated to the governor's re-election campaign",
        "tags": ["education", "corruption", "contracting"],
    },
    {
        "event_type": "political_decision",
        "who": "Trade Representative Office",
        "what": "Signed a trade agreement omitting labor protection clauses negotiated for two years",
        "when": "August 1, 2023",
        "where": "Geneva, Switzerland",
        "why": "Presented as 'expediting economic cooperation'",
        "casualties": None,
        "financial_impact": "Estimated $3 billion in tariff reductions favoring imported goods",
        "responsible_party": "Trade Representative Office",
        "cover_up_detail": "Labor protection clauses were removed in a last-minute closed session; no transcript was released",
        "tags": ["trade", "labor", "international"],
    },
    {
        "event_type": "political_decision",
        "who": "City Council of Mercer Heights",
        "what": "Approved rezoning of protected wetlands for private development",
        "when": "December 10, 2023",
        "where": "Mercer Heights, New Jersey",
        "why": "Described as 'economic development opportunity'",
        "casualties": None,
        "financial_impact": "$95 million project value",
        "responsible_party": "City Council of Mercer Heights",
        "cover_up_detail": "Two council members who voted yes held financial stakes in the development company through a shell LLC",
        "tags": ["environment", "corruption", "zoning"],
    },
    # ── corporate_action (5) ─────────────────────────────────────────────
    {
        "event_type": "corporate_action",
        "who": "Vantage Pharma",
        "what": "Recalled 800,000 units of a blood pressure medication after patients reported kidney damage",
        "when": "May 14, 2024",
        "where": "United States",
        "why": "Active ingredient degraded 4x above safe limits due to storage flaw",
        "casualties": None,
        "financial_impact": "$340 million recall cost",
        "responsible_party": "Vantage Pharma",
        "cover_up_detail": "Vantage Pharma delayed the recall by 7 months after internal tests first detected the degradation issue",
        "tags": ["pharma", "recall", "cover-up"],
    },
    {
        "event_type": "corporate_action",
        "who": "DataVault Inc.",
        "what": "Disclosed a breach exposing 9 million customer records",
        "when": "March 3, 2024",
        "where": "San Francisco, California",
        "why": "Unpatched vulnerability in customer portal exploited by attackers",
        "casualties": None,
        "financial_impact": "$67 million in settlements and fines",
        "responsible_party": "DataVault Inc.",
        "cover_up_detail": "DataVault only disclosed after a journalist obtained a copy of the breach data and contacted the company",
        "tags": ["data breach", "privacy", "cover-up"],
    },
    {
        "event_type": "corporate_action",
        "who": "Constellation Foods",
        "what": "Merger with rival FreshPath Organics approved by the FTC",
        "when": "November 17, 2023",
        "where": "Washington D.C.",
        "why": "FTC accepted the company's supply-chain efficiency argument",
        "casualties": None,
        "financial_impact": "$4.8 billion acquisition",
        "responsible_party": "Constellation Foods",
        "cover_up_detail": "An internal FTC analysis showing 76% market concentration in three regions was withheld from the public docket",
        "tags": ["antitrust", "merger", "food"],
    },
    {
        "event_type": "corporate_action",
        "who": "TurboAuto Group",
        "what": "Delayed recalling 1.2 million vehicles with defective brake sensors",
        "when": "July 30, 2024",
        "where": "United States and Canada",
        "why": "Sensor firmware error caused brake lights to fail at highway speed",
        "casualties": 4,
        "financial_impact": "$510 million recall and litigation",
        "responsible_party": "TurboAuto Group",
        "cover_up_detail": "TurboAuto's engineering team flagged the defect 9 months before the recall; executives delayed disclosure pending a sales quarter close",
        "tags": ["automotive", "recall", "cover-up", "safety"],
    },
    {
        "event_type": "corporate_action",
        "who": "BlueSky Insurance",
        "what": "Systematically denied claims for storm damage in flood-prone zip codes",
        "when": "2023–2024",
        "where": "Louisiana and Mississippi",
        "why": "Internal algorithm flagged high-risk zip codes for automatic review and denial",
        "casualties": None,
        "financial_impact": "$210 million in withheld payouts",
        "responsible_party": "BlueSky Insurance",
        "cover_up_detail": "The denial algorithm was created after an executive memo stating 'we need to get ahead of climate exposure'",
        "tags": ["insurance", "climate", "discrimination"],
    },
    # ── public_health (5) ────────────────────────────────────────────────
    {
        "event_type": "public_health",
        "who": "Millbrook Municipal Water Authority",
        "what": "Lead levels in tap water exceeded EPA limits in 14 schools and 8 daycares",
        "when": "August–October 2023",
        "where": "Millbrook, Michigan",
        "why": "Aging lead service lines leached into the distribution system",
        "casualties": None,
        "financial_impact": "$38 million remediation",
        "responsible_party": "Millbrook Municipal Water Authority",
        "cover_up_detail": "The Authority delayed public notification by 60 days while internally negotiating a low-interest remediation loan",
        "tags": ["water", "public health", "cover-up"],
    },
    {
        "event_type": "public_health",
        "who": "GeneMed Therapeutics",
        "what": "A clinical trial for an arthritis drug showed elevated cardiac risk in 12% of subjects",
        "when": "April 2024",
        "where": "Phase III trial, United States and Europe",
        "why": "Drug activates a pathway that increases platelet aggregation",
        "casualties": None,
        "financial_impact": "$1.2 billion drug program value",
        "responsible_party": "GeneMed Therapeutics",
        "cover_up_detail": "The cardiac risk finding appeared in trial subgroup data that was omitted from the summary GeneMed submitted to the FDA",
        "tags": ["pharma", "clinical trial", "FDA", "cover-up"],
    },
    {
        "event_type": "public_health",
        "who": "ClearSpring Beverage Co.",
        "what": "Sodium benzoate levels in a popular sports drink were found to exceed safe limits",
        "when": "January 20, 2024",
        "where": "United States",
        "why": "Production batch error resulted in 3x the intended additive concentration",
        "casualties": None,
        "financial_impact": "$22 million in recall and reformulation",
        "responsible_party": "ClearSpring Beverage Co.",
        "cover_up_detail": "The safety study cited by ClearSpring in its FDA filing was funded entirely by ClearSpring and has not been independently replicated",
        "tags": ["food additive", "public health", "conflict of interest"],
    },
    {
        "event_type": "public_health",
        "who": "Hargrove County Health Department",
        "what": "Salmonella outbreak linked to a local egg producer sickened 193 people",
        "when": "October 2023",
        "where": "Hargrove County, Georgia",
        "why": "Unsanitary conditions in poultry housing allowed Salmonella to spread",
        "casualties": 2,
        "financial_impact": "$4.1 million in healthcare costs and settlements",
        "responsible_party": "Hargrove County Health Department",
        "cover_up_detail": "Health department inspectors were instructed by the county administrator to cite only minor violations during the investigation period",
        "tags": ["food safety", "public health", "cover-up"],
    },
    {
        "event_type": "public_health",
        "who": "NeuralPath Biotech",
        "what": "Published results claiming its experimental Alzheimer's drug slowed cognitive decline by 38%",
        "when": "March 2024",
        "where": "Journal of Neuroscience, United States",
        "why": "Drug targets amyloid plaque accumulation",
        "casualties": None,
        "financial_impact": "$4 billion in projected market value",
        "responsible_party": "NeuralPath Biotech",
        "cover_up_detail": "Imaging data from 24% of trial participants was excluded post-hoc without disclosure; including it reduces efficacy to 11%",
        "tags": ["pharma", "clinical trial", "research integrity"],
    },
]

# ---------------------------------------------------------------------------
# Bias profiles per framing archetype
# ---------------------------------------------------------------------------

_BIAS_POOL = [
    BiasProfile(
        source_name="The National Tribune",
        omitted_fields=["responsible_party", "cover_up_detail"],
        framing="pro_corporate",
        credibility_hint=True,
    ),
    BiasProfile(
        source_name="GovWatch Monitor",
        omitted_fields=["financial_impact", "casualties"],
        framing="pro_government",
        credibility_hint=True,
    ),
    BiasProfile(
        source_name="The Daily Herald",
        omitted_fields=["why", "cover_up_detail"],
        framing="neutral",
        credibility_hint=True,
    ),
    BiasProfile(
        source_name="TruthPulse",
        omitted_fields=["responsible_party", "financial_impact", "why"],
        framing="sensationalist",
        credibility_hint=False,
    ),
    BiasProfile(
        source_name="CorpScope",
        omitted_fields=["cover_up_detail", "casualties", "why"],
        framing="pro_corporate",
        credibility_hint=False,
    ),
    BiasProfile(
        source_name="Public Eye Weekly",
        omitted_fields=["financial_impact"],
        framing="neutral",
        credibility_hint=True,
    ),
]

# ---------------------------------------------------------------------------
# Article rendering templates
# ---------------------------------------------------------------------------

def _render_article(event: EventGroundTruth, profile: BiasProfile) -> str:
    """Render a news article from ground truth using the given bias profile."""
    omit = set(profile.omitted_fields)

    responsible = (
        "" if "responsible_party" in omit
        else f"{event.responsible_party} has been identified as the responsible party. "
    )
    casualties_line = ""
    if event.casualties and "casualties" not in omit:
        casualties_line = f"{event.casualties} people were affected. "
    why_line = (
        "" if "why" in omit
        else f"Investigators attributed the incident to: {event.why}. "
    )
    financial_line = (
        "" if "financial_impact" in omit or not event.financial_impact
        else f"Financial impact is estimated at {event.financial_impact}. "
    )
    cover_up_line = (
        "" if "cover_up_detail" in omit
        else f"Furthermore, {event.cover_up_detail}. "
    )

    if profile.framing == "pro_corporate":
        opening = (
            f"{profile.source_name} has learned that a significant operational development "
            f"occurred at {event.where} on {event.when}. {event.what}. "
            f"The company has stated it is cooperating fully with investigators. "
        )
    elif profile.framing == "pro_government":
        opening = (
            f"According to {profile.source_name}, an incident was reported at "
            f"{event.where} on {event.when}. {event.what}. "
            f"Authorities confirmed they are managing the situation appropriately. "
        )
    elif profile.framing == "sensationalist":
        opening = (
            f"BREAKING — {profile.source_name}: Chaos erupted at {event.where} "
            f"on {event.when}! {event.what}. Sources say the situation is dire. "
        )
    else:  # neutral
        opening = (
            f"{profile.source_name} reports that an incident occurred at "
            f"{event.where} on {event.when}. {event.what}. "
            f"Authorities have been notified. "
        )

    return (opening + responsible + casualties_line + why_line + financial_line + cover_up_line).strip()


# ---------------------------------------------------------------------------
# EventGenerator
# ---------------------------------------------------------------------------

class EventGenerator:
    """
    Generates and caches 200 synthetic events with articles.
    Fully deterministic given the same seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._events: List[Tuple[EventGroundTruth, Dict[str, dict]]] = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        """Pre-generate 200 events with 4 articles each."""
        templates = _EVENT_TEMPLATES  # 20 templates
        for i in range(200):
            tmpl = templates[i % len(templates)]
            event = EventGroundTruth(
                event_id=f"evt_{i:04d}",
                event_type=tmpl["event_type"],
                who=tmpl["who"],
                what=tmpl["what"],
                when=tmpl["when"],
                where=tmpl["where"],
                why=tmpl["why"],
                casualties=tmpl.get("casualties"),
                financial_impact=tmpl.get("financial_impact"),
                responsible_party=tmpl["responsible_party"],
                cover_up_detail=tmpl["cover_up_detail"],
                red_herring_source_id=None,  # set later for hard tasks
                tags=list(tmpl["tags"]),
            )
            articles = self._make_articles(event, n=4)
            self._events.append((event, articles))

    def _make_articles(
        self, event: EventGroundTruth, n: int
    ) -> Dict[str, dict]:
        """Build n articles ensuring cover_up_detail is omitted by at least 2 sources."""
        profiles = self._rng.sample(_BIAS_POOL, k=min(n, len(_BIAS_POOL)))
        # Guarantee at least 2 omit cover_up_detail
        profiles_with_cover = [p for p in profiles if "cover_up_detail" in p.omitted_fields]
        if len(profiles_with_cover) < 2:
            # Force first profile to omit cover_up_detail
            profiles[0] = BiasProfile(
                source_name=profiles[0].source_name,
                omitted_fields=list(set(profiles[0].omitted_fields) | {"cover_up_detail"}),
                framing=profiles[0].framing,
                credibility_hint=profiles[0].credibility_hint,
            )

        articles: Dict[str, dict] = {}
        for idx, profile in enumerate(profiles[:n]):
            src_id = f"src_{idx}"
            text = _render_article(event, profile)
            articles[src_id] = {
                "source_name": profile.source_name,
                "framing": profile.framing,
                "credibility_hint": profile.credibility_hint,
                "omitted_fields": list(profile.omitted_fields),
                "text": text,
                "fields": {
                    "responsible_party": (
                        None if "responsible_party" in profile.omitted_fields
                        else event.responsible_party
                    ),
                    "cover_up_detail": (
                        None if "cover_up_detail" in profile.omitted_fields
                        else event.cover_up_detail
                    ),
                    "financial_impact": (
                        None if "financial_impact" in profile.omitted_fields
                        else event.financial_impact
                    ),
                    "casualties": (
                        None if "casualties" in profile.omitted_fields
                        else event.casualties
                    ),
                    "why": (
                        None if "why" in profile.omitted_fields
                        else event.why
                    ),
                },
            }
        return articles

    def sample(
        self, task_name: str
    ) -> Tuple[EventGroundTruth, Dict[str, dict]]:
        """Sample a random event appropriate for the given task."""
        try:
            from .tasks import TASK_CONFIG
        except ImportError:
            from server.tasks import TASK_CONFIG
        config = TASK_CONFIG[task_name]
        num_sources = config["num_sources"]
        has_red_herring = config.get("has_red_herring", False)

        event, all_articles = self._rng.choice(self._events)
        # Slice to num_sources
        source_ids = list(all_articles.keys())[:num_sources]
        articles = {sid: all_articles[sid] for sid in source_ids}

        # For hard tasks, designate a red herring source
        if has_red_herring and len(source_ids) >= 2:
            rh_id = self._rng.choice(source_ids[1:])  # never first source
            import copy
            event = copy.copy(event)
            event.red_herring_source_id = rh_id

        return event, articles
