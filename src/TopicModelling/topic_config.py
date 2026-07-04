"""
Path config for src/TopicModelling/.
Assumes this file lives at: <project_root>/src/TopicModelling/topic_config.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../DSSE_ASSIGNMENT_3

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

ONTOLOGY_XLSX = DATA_DIR / "ontology_sheet_ref.xlsx"
PROCESSED_ISSUES_CSV = OUTPUT_DIR / "processed_issues.csv"   # from 02_preprocess.py

# everything this sub-pipeline produces goes here, so it doesn't clobber
# week 1's output/ files
TM_OUTPUT_DIR = OUTPUT_DIR / "topic_modelling"
ONTOLOGY_MAP_JSON = TM_OUTPUT_DIR / "ontology_term_map.json"
PROCESSED_ONTOLOGY_CSV = TM_OUTPUT_DIR / "processed_issues_ontology.csv"

LDA_BASELINE_DIR = TM_OUTPUT_DIR / "lda_baseline"
LDA_ONTOLOGY_DIR = TM_OUTPUT_DIR / "lda_ontology"
BERTOPIC_DIR = TM_OUTPUT_DIR / "bertopic"

# ---- LDA parameters ----
# Iteration 1: many topics, low alpha/beta -> check if project-specific noise dominates
ITER1_NUM_TOPICS = 5   # set from the coherence sweep result - change if you rerun coherence
ITER1_ALPHA = 0.01
ITER1_BETA = 0.01

ITER2_NUM_TOPICS = 10   # set from the coherence sweep result on ontology tokens
ITER2_ALPHA = 0.01
ITER2_BETA = 0.01

TOP_N_WORDS = 15
MIN_DF = 5
MAX_DF = 0.6

# tokens that are pipeline artifacts, not real vocabulary - drop before LDA
EXTRA_STOPWORDS = {"simpleclassname", "versionnumber", "filepath", "noformatblock"}

# ---- Ontology classes, per the assignment text (NOT the raw column names in
# ontology_sheet_ref.xlsx - that file is only the "further examples" sheet the
# assignment links to; its column headers don't match the assignment's own
# class names, so we translate them here) ----
#
# Assignment's classes:
#   Component  - processing units at any abstraction (machine, service, class, method)
#   Connector  - communication between components, usually verbs (send, write, retrieve)
#   Data       - data stored/transferred (message, object, dump)
#   Solution   - named patterns/tactics (layer, MVC, replication, authentication)
#   Quality attribute - performance, security, etc. (optionally split per-attribute)
#
# The sheet's "Unnamed: 0" column (method/procedure/interface/field/...) is also
# component-level vocabulary, so it's folded into COMPONENT alongside "Component_".
# "Connector_Data_" holds nouns (socket, payload, message, dump) -> that's DATA,
# not a connector. "Connector_" holds verbs (retrieve, send, write) -> CONNECTOR.
# "Pattern_" holds pattern/tactic names -> SOLUTION.
# "Technology_" (framework/protocol/product names) and "Requirement_" plus the
# two trailing unnamed columns are outside the assignment's ontology entirely
# (they're generic web-app/domain vocabulary from the source repo's own
# example project) - excluded by default; Technology_ can be opted back in
# since "technology upgrade" is one of the assignment's example useful topics.
ONTOLOGY_COLUMN_TO_CLASS = {
    "Unnamed: 0": "COMPONENT",
    "Component_": "COMPONENT",
    "Connector_Data_": "DATA",
    "Connector_": "CONNECTOR",
    "Pattern_": "SOLUTION",
}
INCLUDE_TECHNOLOGY_CLASS = True
if INCLUDE_TECHNOLOGY_CLASS:
    ONTOLOGY_COLUMN_TO_CLASS["Technology_"] = "TECHNOLOGY"

QUALITY_ATTRIBUTE_COLUMN = "Quality_Attribute_"

# Per the assignment: "One option [is] that you create separate classes for
# each quality attribute. This might produce more detailed topics."
# The sheet lists ~145 individual QA terms (noun/adjective word-forms of the
# same attribute, e.g. "security"/"secure"/"securability") with no grouping,
# so we curate the common ones into canonical buckets ourselves. Any QA term
# NOT in this dict is left as its own token (unreplaced) rather than dumped
# into a catch-all - per the assignment's "only replace terms you're sure
# about", a term we haven't deliberately grouped shouldn't be merged blindly.
SPLIT_QUALITY_ATTRIBUTES = True
QUALITY_ATTRIBUTE_GROUPS = {
    "QA_SECURITY": ["security", "secure", "securability"],
    "QA_PERFORMANCE": ["efficiency", "efficient", "speed", "throughput", "latency", "responsiveness"],
    "QA_RELIABILITY": ["reliability", "reliable", "reliably", "robustness", "robust",
                        "durability", "durable", "recoverability", "recoverable",
                        "resilience", "stability", "stable", "survivability"],
    "QA_AVAILABILITY": ["availability", "available"],
    "QA_SCALABILITY": ["scalability", "scale", "scalable", "scaling"],
    "QA_USABILITY": ["usability", "usable", "learnability", "simplicity", "simple",
                      "understandability", "understable"],
    "QA_MAINTAINABILITY": ["maintainability", "maintainable", "maintenance",
                            "modifiability", "modifiable", "serviceability",
                            "supportability", "debuggability", "debuggable",
                            "testability", "testable"],
    "QA_PORTABILITY": ["portability", "portable", "interoperability", "interoperable",
                        "compatibility", "compatible", "adaptability", "adaptable",
                        "adapte", "configurability", "configurable", "customizability",
                        "customizable", "tailorability", "tailorable", "tailor"],
    "QA_SAFETY": ["safety", "safe"],
    "QA_ACCESSIBILITY": ["accessibility", "accessible"],
    "QA_MODULARITY": ["modularity", "extensibility", "extensible", "flexibility",
                       "flexible", "composability", "composable", "orthogonality"],
    "QA_ACCOUNTABILITY": ["accountability", "accountable", "auditability", "auditable",
                           "traceability", "traceable"],
    "QA_CORRECTNESS": ["correctness", "accuracy", "accurate", "precision", "fidelity",
                        "integrity", "integrate"],
}

MIN_FREQ_TO_REPLACE = 40  # raised from 20 - iteration 1 showed 20 was too aggressive