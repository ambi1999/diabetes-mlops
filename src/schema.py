"""The data contract: which features exist and which values are allowed.

CONCEPT - single source of truth: data validation, the API's input checks and
drift monitoring all import this one definition. If training and serving each
had their own copy, they could silently drift apart ("training/serving skew").

These are fixed facts about the BRFSS 2015 survey, not tunable settings,
which is why they live here rather than in config.py."""

TARGET = "Diabetes_binary"  # 0 = no diabetes, 1 = prediabetes or diabetes

# feature name -> (min allowed, max allowed). Codes follow the BRFSS 2015 survey.
FEATURE_RANGES = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),  # cholesterol check in last 5 years
    "BMI": (12, 98),
    "Smoker": (0, 1),  # smoked 100+ cigarettes in lifetime
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),  # exercise in past 30 days
    "Fruits": (0, 1),  # fruit 1+ times per day
    "Veggies": (0, 1),  # vegetables 1+ times per day
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),  # couldn't see a doctor because of cost
    "GenHlth": (1, 5),  # 1 = excellent ... 5 = poor
    "MentHlth": (0, 30),  # days of poor mental health, past 30
    "PhysHlth": (0, 30),  # days of poor physical health, past 30
    "DiffWalk": (0, 1),  # difficulty walking or climbing stairs
    "Sex": (0, 1),  # 0 = female, 1 = male
    "Age": (1, 13),  # 1 = 18-24, 2 = 25-29 ... 13 = 80+
    "Education": (1, 6),  # 1 = never attended ... 6 = college graduate
    "Income": (1, 8),  # 1 = < $10k ... 8 = $75k+
}

FEATURES = list(FEATURE_RANGES)
CONTINUOUS_FEATURES = ["BMI"]  # every other feature is a category or a code

# One realistic, made-up person, defined ONCE and reused everywhere an example
# input is needed: the API's interactive docs (/docs), the API tests, the
# smoke test and the demo notebooks. If the contract above changes, only this
# example needs updating (single source of truth).
EXAMPLE_INPUT = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 33.0,
    "Smoker": 1,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits": 0,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 4,
    "MentHlth": 5,
    "PhysHlth": 10,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 10,
    "Education": 4,
    "Income": 5,
}
