# mapping.py

SYMPTOM_TO_ICD = {
    "UTI symptoms (burning/frequency/urgency)": ["icd_infect_1", "icd_infect_2"],
    "Skin infection / cellulitis symptoms": ["icd_infect_5", "icd_misc_10"],
    "Fever / chills (infection-like)": ["icd_infect_3"],
    "Numbness / tingling (neuropathy-like)": ["icd_neuro_1", "icd_neuro_3"],
    "Blurred vision / eye issues": ["icd_eye_1", "icd_eye_2"],
    "Swelling / kidney concerns": ["icd_renal_1", "icd_renal_4"],
    "Chest pain / known cardiovascular history": ["icd_cardio_1", "icd_cardio_6"],
    "Dehydration / electrolyte issues": ["icd_electrolyte_1", "icd_electrolyte_2"],
}

MED_OPTIONS = [
    "med_insulin",
    "med_metformin",
    "med_sulfonylureas",
    "med_ace_inhibitor",
    "med_beta_blocker",
    "med_statins",
    "med_thiazolidinediones",
    "med_dpp4_inhibitor",
    "med_sglt2_inhibitor",
    "med_glp1_agonist",
    "med_nsaid",
    "med_antibiotic",
    "med_anticoagulant",
    "med_diuretic",
    "med_opioid",
    "med_antidepressant",
    "med_ppi",
    "med_antihistamine",
    "med_other",
]

def build_feature_dict(age, gender, selected_meds, selected_symptoms):
    x = {"age": int(age), "gender": 1 if gender.lower() == "male" else 0}

    for m in selected_meds:
        x[m] = 1

    for s in selected_symptoms:
        for icd_feat in SYMPTOM_TO_ICD.get(s, []):
            x[icd_feat] = 1

    return x
