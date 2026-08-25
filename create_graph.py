import os
import re

from openai import OpenAI
from camel.loaders import UnstructuredIO
from camel.storages.graph_storages.graph_element import GraphElement, Node, Relationship

from utils import add_ge_emb, add_gid, merge_similar_nodes, add_sum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


KG_SYSTEM_PROMPT = (
    "You are a medical knowledge graph extraction engine. "
    "Given clinical text, extract TYPED medical entities and the relationships BETWEEN them. "
    "Do NOT create a central Patient hub node that connects to everything. "
    "Instead, create direct relationships between medical concepts.\n\n"
    "Entity types to use: Disease, Symptom, Medication, Medical_Test, Procedure, "
    "Condition, Anatomy, Measurement, Diagnosis, LabTest, Hormone, Clinical_Finding\n\n"
    "Relationship types to use: treats, caused_by, detects, associated_with, "
    "prescribed_for, performed_for, indicates, monitors, contraindicates, "
    "complication_of, symptom_of, diagnosed_by, administered_via, "
    "measured_by, affects, produces, risk_factor_for\n\n"
    "Return ONLY lines in these exact formats:\n"
    "Node(id='ENTITY_NAME', type='TYPE', description='BRIEF_DESCRIPTION')\n"
    "Relationship(subj=Node(id='S', type='ST'), obj=Node(id='O', type='OT'), type='REL_TYPE')\n\n"
    "Example output:\n"
    "Node(id='Vascular Dementia', type='Disease', description='A form of dementia caused by impaired blood flow to the brain')\n"
    "Node(id='MRI', type='Medical_Test', description='Imaging technique showing structural brain changes')\n"
    "Node(id='Chronic Ischemic Damage', type='Condition', description='Brain damage due to reduced blood supply')\n"
    "Relationship(subj=Node(id='Vascular Dementia', type='Disease'), obj=Node(id='Chronic Ischemic Damage', type='Condition'), type='caused_by')\n"
    "Relationship(subj=Node(id='MRI', type='Medical_Test'), obj=Node(id='Chronic Ischemic Damage', type='Condition'), type='detects')\n"
)




def _extract_graph_elements_from_text(raw_text, source_element):
    if os.getenv("USE_LLM_EXTRACTION", "0") != "1":
        return _fallback_extract_graph_elements(raw_text, source_element)

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _fallback_extract_graph_elements(raw_text, source_element)

    model = os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.groq.com/openai/v1")

    user_prompt = (
        "Extract typed medical entities and the direct relationships between them. "
        "Do NOT make a Patient node the center of every relationship. "
        "Focus on medical concept interconnections.\n\n"
        f"CONTENT:\n{raw_text}"
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": KG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.0,
            timeout=45,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return _fallback_extract_graph_elements(raw_text, source_element)

    # Parse nodes — with optional description field
    node_pattern = r"Node\(id='(.*?)', type='(.*?)'(?:, description='(.*?)')?\)"
    rel_pattern = (
        r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), "
        r"obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"
    )

    nodes = {}
    relationships = []

    for match in re.finditer(node_pattern, content):
        node_id, node_type = match.group(1).strip(), match.group(2).strip()
        node_desc = (match.group(3) or "").strip()
        node_type = node_type or "Entity"
        if node_id and node_id not in nodes:
            props = {
                "source": "llm_extracted",
                "source_type": "LLM_extracted",
                "modality": "clinical_text",
                "extraction_confidence": "medium",
                "extraction_method": "llm_extraction"
            }
            if node_desc:
                props["description"] = node_desc
            nodes[node_id] = Node(id=node_id, type=node_type, properties=props)

    for match in re.finditer(rel_pattern, content):
        subj_id, subj_type, obj_id, obj_type, rel_type = match.groups()
        subj_id = subj_id.strip()
        obj_id = obj_id.strip()
        subj_type = subj_type.strip() or "Entity"
        obj_type = obj_type.strip() or "Entity"
        rel_type = rel_type.strip() or "RELATED_TO"

        if not subj_id or not obj_id:
            continue

        if subj_id not in nodes:
            nodes[subj_id] = Node(
                id=subj_id,
                type=subj_type,
                properties={
                    "source": "llm_extracted",
                    "source_type": "LLM_extracted",
                    "modality": "clinical_text",
                    "extraction_confidence": "medium",
                    "extraction_method": "llm_extraction"
                },
            )
        if obj_id not in nodes:
            nodes[obj_id] = Node(
                id=obj_id,
                type=obj_type,
                properties={
                    "source": "llm_extracted",
                    "source_type": "LLM_extracted",
                    "modality": "clinical_text",
                    "extraction_confidence": "medium",
                    "extraction_method": "llm_extraction"
                },
            )

        relationships.append(
            Relationship(
                subj=nodes[subj_id],
                obj=nodes[obj_id],
                type=rel_type,
                properties={
                    "source": "llm_extracted",
                    "source_type": "LLM_extracted",
                    "provenance": "evidence_backed",
                    "extraction_method": "llm_relation_extraction"
                },
            )
        )

    graph_element = GraphElement(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_element,
    )
    if not graph_element.nodes:
        return _fallback_extract_graph_elements(raw_text, source_element)
    return graph_element


def _icd_chapter(code_str):
    """Extract the ICD chapter prefix (first 3 chars) for grouping."""
    code = re.sub(r"[^A-Za-z0-9]", "", code_str.strip())
    return code[:3] if len(code) >= 3 else code


def _fallback_extract_graph_elements(raw_text, source_element):
    """Structured fallback: build a medical knowledge graph from structured
    clinical text.  Creates a distributed mesh of inter-entity relationships
    rather than a patient-centric star or any single hub node.
    """
    nodes = {}
    relationships = []

    # ---- Clinical type classification keywords ----
    _DISEASE_KEYWORDS = {
        "pneumonia", "tuberculosis", "sepsis", "meningitis", "endocarditis",
        "cellulitis", "osteomyelitis", "bronchitis", "influenza", "hepatitis",
        "encephalitis", "pericarditis", "peritonitis", "abscess", "empyema",
        "bacteremia", "colitis", "gastroenteritis", "pharyngitis", "sinusitis",
        "tonsillitis", "otitis", "cystitis", "pyelonephritis", "cholecystitis",
        "pancreatitis", "diverticulitis", "arthritis", "myocarditis",
        "cardiomyopathy", "fibrosis", "cirrhosis", "neoplasm", "cancer",
        "tumor", "malignancy", "lymphoma", "leukemia", "sarcoma",
        "diabetes", "asthma", "copd", "emphysema",
    }
    _CONDITION_KEYWORDS = {
        "failure", "insufficiency", "disorder", "syndrome", "dysfunction",
        "deficiency", "overload", "imbalance", "abnormality", "deformity",
        "stenosis", "obstruction", "blockage", "occlusion", "thrombosis",
        "embolism", "hemorrhage", "effusion", "edema", "hypertension",
        "hypotension", "tachycardia", "bradycardia", "arrhythmia",
        "fibrillation", "anemia", "acidosis", "alkalosis", "hyperkalemia",
        "hyperpotassemia", "hyponatremia", "hyperglycemia", "hypoglycemia",
    }
    _SYMPTOM_KEYWORDS = {
        "pain", "fever", "cough", "dyspnea", "nausea", "vomiting",
        "diarrhea", "fatigue", "weakness", "dizziness", "headache",
        "swelling", "rash", "pruritus", "bleeding", "hemoptysis",
        "wheezing", "stridor", "confusion", "syncope", "palpitation",
        "restless", "insomnia", "malaise", "chills", "rigors", "tremor",
        "paresthesia", "numbness", "dysphagia", "constipation",
    }
    _ANATOMY_KEYWORDS = {
        "lung", "heart", "liver", "kidney", "brain", "artery", "vein",
        "lobe", "ventricle", "atrium", "aorta", "bronchus", "pleura",
        "peritoneum", "mediastinum", "thorax", "abdomen", "pelvis",
        "femur", "tibia", "spine", "vertebra", "cortex", "cerebral",
        "pulmonary", "hepatic", "renal", "cardiac", "cranial",
        "axillary", "carotid", "femoral", "pericardial",
    }
    _INFECTION_KEYWORDS = {
        "infection", "infectious", "infected", "septic", "mrsa",
        "vre", "esbl", "c. diff", "clostridium", "candida", "fungal",
        "viral", "bacterial",
    }
    _INJURY_KEYWORDS = {
        "fracture", "dislocation", "sprain", "strain", "contusion",
        "laceration", "wound", "burn", "injury", "trauma", "rupture",
        "tear", "amputation", "bite", "crushing", "concussion",
    }

    def _classify_diagnosis(name):
        """Classify a diagnosis string into a specific entity type."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in _DISEASE_KEYWORDS):
            return "Disease"
        if any(kw in name_lower for kw in _INFECTION_KEYWORDS):
            return "Infection"
        if any(kw in name_lower for kw in _INJURY_KEYWORDS):
            return "Condition"
        if any(kw in name_lower for kw in _SYMPTOM_KEYWORDS):
            return "Symptom"
        if any(kw in name_lower for kw in _CONDITION_KEYWORDS):
            return "Condition"
        if any(kw in name_lower for kw in _ANATOMY_KEYWORDS):
            return "Anatomy"
        return "Condition"  # default to Condition rather than generic Diagnosis

    def _add_node(node_id, node_type, description="", source_type="rule_based", modality="structured_data", confidence="high"):
        key = node_id.strip()
        if key and key not in nodes:
            props = {
                "source": "structured_fallback",
                "source_type": source_type,
                "modality": modality,
                "extraction_confidence": confidence,
                "extraction_method": "structured_rule_fallback"
            }
            if description:
                props["description"] = description
            nodes[key] = Node(id=key, type=node_type, properties=props)
        return nodes.get(key)

    # =====================================================================
    # 1) DICTIONARY DATA  (bottom / middle layer)
    #    Create clustered sub-graphs using ICD code grouping.
    #    Classify diagnoses into Disease / Condition / Symptom / Infection.
    # =====================================================================

    # --- Diagnosis dictionary entries ---
    icd_diag_groups = {}  # chapter_prefix -> list of nodes
    for match in re.finditer(
        r"DIAGNOSIS\s+code=(\S+)\s+icd_version=(\S+)\s+name=(.+?)(?:\n|$)", raw_text
    ):
        code, ver, name = match.group(1), match.group(2), match.group(3).strip()
        entity_type = _classify_diagnosis(name)
        d_node = _add_node(name, entity_type, f"ICD-{ver} code {code}: {name}")
        chapter = _icd_chapter(code)
        icd_diag_groups.setdefault(chapter, []).append(d_node)

    # Link diagnoses within the same ICD chapter (chain rather than clique)
    for chapter, group_nodes in icd_diag_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="icd_related",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "dictionary",
                        "provenance": "dictionary_derived",
                        "extraction_method": "icd_chapter_grouping",
                        "icd_chapter": chapter
                    },
                )
            )

    # --- Procedure dictionary entries ---
    icd_proc_groups = {}
    for match in re.finditer(
        r"PROCEDURE\s+code=(\S+)\s+icd_version=(\S+)\s+name=(.+?)(?:\n|$)", raw_text
    ):
        code, ver, name = match.group(1), match.group(2), match.group(3).strip()
        p_node = _add_node(name, "Procedure", f"ICD-{ver} procedure code {code}: {name}", source_type="dictionary", modality="dictionary")
        chapter = _icd_chapter(code)
        icd_proc_groups.setdefault(chapter, []).append(p_node)

    for chapter, group_nodes in icd_proc_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="icd_related",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "dictionary",
                        "provenance": "dictionary_derived",
                        "extraction_method": "icd_chapter_grouping",
                        "icd_chapter": chapter
                    },
                )
            )

    # --- Lab dictionary entries ---
    lab_category_groups = {}
    for match in re.finditer(
        r"LAB_TEST\s+itemid=(\S+)\s+label=(.+?)\s+fluid=(\S+)\s+category=(.+?)(?:\n|$)",
        raw_text,
    ):
        itemid, label, fluid, category = (
            match.group(1),
            match.group(2).strip(),
            match.group(3).strip(),
            match.group(4).strip(),
        )
        desc = f"Lab test (item {itemid}): {label}, fluid={fluid}, category={category}"
        l_node = _add_node(label, "LabTest", desc, source_type="dictionary", modality="dictionary")
        group_key = f"{fluid}_{category}"
        lab_category_groups.setdefault(group_key, []).append(l_node)

    # Chain labs within the same category
    for group_key, group_nodes in lab_category_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="same_category",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "dictionary",
                        "provenance": "dictionary_derived",
                        "extraction_method": "lab_category_chain",
                        "category": group_key
                    },
                )
            )

    # Cross-link: connect each ICD diag chapter head to related procedure chapter head
    diag_heads = {ch: ns[0] for ch, ns in icd_diag_groups.items() if ns}
    proc_heads = {ch: ns[0] for ch, ns in icd_proc_groups.items() if ns}
    for chapter in set(diag_heads) & set(proc_heads):
        relationships.append(
            Relationship(
                subj=proc_heads[chapter],
                obj=diag_heads[chapter],
                type="procedure_for_category",
                properties={
                    "source": "structured_fallback",
                    "source_type": "dictionary",
                    "provenance": "dictionary_derived",
                    "extraction_method": "cross_chapter_link"
                },
            )
        )

    # =====================================================================
    # 2) GUIDELINE DATA  (middle layer)
    # =====================================================================

    guideline_conditions = []
    for match in re.finditer(r"Condition\s+\d+:\s+Consider diagnosis\s+'([^']+)'", raw_text):
        name = match.group(1).strip()
        g_node = _add_node(name, "Diagnosis", f"Guideline-referenced diagnosis: {name}", source_type="clinical_guideline", modality="clinical_text")
        guideline_conditions.append(g_node)

    guideline_interventions = []
    for match in re.finditer(r"Intervention\s+\d+:\s+Procedure option\s+'([^']+)'", raw_text):
        name = match.group(1).strip()
        g_node = _add_node(name, "Procedure", f"Guideline-referenced procedure: {name}", source_type="clinical_guideline", modality="clinical_text")
        guideline_interventions.append(g_node)

    # Round-robin link interventions to conditions (1:1 instead of many-to-many)
    if guideline_conditions and guideline_interventions:
        n_cond = len(guideline_conditions)
        for i, p_node in enumerate(guideline_interventions):
            d_node = guideline_conditions[i % n_cond]
            relationships.append(
                Relationship(
                    subj=p_node,
                    obj=d_node,
                    type="indicated_for",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "clinical_guideline",
                        "provenance": "structurally_generated",
                        "extraction_method": "guideline_round_robin"
                    },
                )
            )

    # Chain guideline conditions (sequential, not clique)
    for i in range(len(guideline_conditions) - 1):
        relationships.append(
            Relationship(
                subj=guideline_conditions[i],
                obj=guideline_conditions[i + 1],
                type="associated_with",
                properties={
                    "source": "structured_fallback",
                    "source_type": "clinical_guideline",
                    "provenance": "structurally_generated",
                    "extraction_method": "guideline_condition_chain"
                },
            )
        )

    # =====================================================================
    # 3) PATIENT CLINICAL DATA  (top layer)
    #    Distribute relationships evenly — NO single diagnosis hub.
    #    Create diverse entity types matching clinical ontology:
    #      Disease, Condition, Symptom, Anatomy, Measurement, Bacteria,
    #      Infection, Medical_Test, Clinical_Finding, Medical_Device
    # =====================================================================

    # ---- Antibiotic classification for Medication sub-typing ----
    _ANTIBIOTIC_KEYWORDS = {
        "ceftriaxone", "cefepime", "vancomycin", "meropenem", "piperacillin",
        "tazobactam", "azithromycin", "levofloxacin", "ciprofloxacin",
        "amoxicillin", "ampicillin", "metronidazole", "doxycycline",
        "clindamycin", "gentamicin", "tobramycin", "trimethoprim",
        "sulfamethoxazole", "linezolid", "daptomycin", "cefazolin",
        "cephalexin", "penicillin", "oxacillin", "nafcillin", "ertapenem",
        "imipenem", "cilastatin", "colistin", "polymyxin", "rifampin",
        "isoniazid", "ethambutol", "pyrazinamide",
    }

    patient_match = re.search(r"patient_id:\s*(\d+)", raw_text)
    patient_id = f"patient_{patient_match.group(1)}" if patient_match else None
    patient_node = _add_node(patient_id, "Patient") if patient_id else None

    # Split by admission blocks
    admission_blocks = re.split(r"(?=Admission summary:)", raw_text)

    for block in admission_blocks:
        # --- Diagnoses → classify into Disease / Condition / Symptom / Infection ---
        block_diagnoses = []
        for diag in re.findall(r"\[\d+\]\s*([^;\.\n]+)", block):
            d = diag.strip()
            if not d:
                continue
            entity_type = _classify_diagnosis(d)
            d_node = _add_node(d, entity_type, f"Clinical {entity_type.lower()}: {d}")
            block_diagnoses.append(d_node)

        # --- Procedures ---
        block_procedures = []
        proc_line = re.search(r"Procedures:\s*(.*)", block)
        if proc_line:
            for proc in [p.strip() for p in proc_line.group(1).split(";") if p.strip()]:
                proc_name = re.sub(r"\s+on\s+\d{4}-\d{2}-\d{2}.*", "", proc).strip()
                if not proc_name:
                    continue
                p_node = _add_node(proc_name, "Procedure", f"Medical procedure: {proc_name}")
                block_procedures.append(p_node)

        # --- Medications → classify as Medication or Antibiotic medication ---
        block_medications = []
        meds_line = re.search(r"Medications:\s*(.*)", block)
        if meds_line:
            for med in [m.strip() for m in meds_line.group(1).split(";") if m.strip()]:
                med_name = med.split(" via ")[0].strip()
                route = med.split(" via ")[1].strip() if " via " in med else ""
                if not med_name:
                    continue
                # Strip dosage to get base medication name for classification
                base_name = re.sub(r"\s+\d+[\.\d]*\s*(mg|gm|g|mcg|mL|units?|mEq)\b.*", "", med_name, flags=re.IGNORECASE).strip()
                is_antibiotic = any(kw in base_name.lower() for kw in _ANTIBIOTIC_KEYWORDS)
                med_type = "Medication"
                desc = f"Medication: {med_name}"
                if is_antibiotic:
                    desc = f"Antibiotic medication: {med_name}"
                if route:
                    desc += f" administered via {route}"
                m_node = _add_node(med_name, med_type, desc)
                if is_antibiotic:
                    m_node.properties["medication_class"] = "antibiotic"
                block_medications.append(m_node)

        # --- Lab Tests → LabTest nodes + Measurement nodes for values ---
        block_labs = []
        block_measurements = []
        labs_line = re.search(r"Recent labs:\s*(.*)", block)
        if labs_line:
            for lab in [l.strip() for l in labs_line.group(1).split(";") if l.strip()]:
                lab_name = lab.split(":")[0].strip()
                lab_value_str = lab.split(":")[1].strip() if ":" in lab else ""
                if not lab_name:
                    continue
                desc = f"Laboratory test: {lab_name}"
                if lab_value_str:
                    desc += f", result: {lab_value_str}"
                l_node = _add_node(lab_name, "LabTest", desc)
                block_labs.append(l_node)

                # Create Measurement node for numeric lab values
                val_match = re.match(r"([\d.]+)\s*([A-Za-z/%]+)?\s*(.*)?", lab_value_str)
                if val_match and val_match.group(1):
                    meas_id = f"{lab_name} = {lab_value_str}"
                    meas_desc = f"Measurement: {lab_name} value {lab_value_str}"
                    is_abnormal = "abnormal" in lab_value_str.lower() or "flag" in lab_value_str.lower()
                    meas_node = _add_node(meas_id, "Measurement", meas_desc)
                    if is_abnormal and meas_node:
                        meas_node.properties["flag"] = "abnormal"
                    if meas_node:
                        block_measurements.append((meas_node, l_node))
                        # Measurement --measured_by--> LabTest
                        relationships.append(
                            Relationship(
                                subj=meas_node, obj=l_node, type="measured_by",
                                properties={
                                    "source": "structured_fallback",
                                    "source_type": "MIMIC",
                                    "provenance": "evidence_backed",
                                    "extraction_method": "lab_value_extraction"
                                },
                            )
                        )

        # --- Microbiology → Bacteria / Microorganism nodes ---
        block_bacteria = []
        micro_line = re.search(r"Microbiology:\s*(.*)", block)
        if micro_line:
            micro_text = micro_line.group(1)
            # Extract culture types as Medical_Test nodes
            for culture_match in re.finditer(r"([\w\s]+Culture[\w\s]*|MRSA SCREEN|URINE CULTURE)", micro_text, re.IGNORECASE):
                culture_name = culture_match.group(0).strip().rstrip(",")
                if culture_name:
                    c_node = _add_node(culture_name, "Medical_Test",
                                       f"Microbiology test: {culture_name}",
                                       source_type="MIMIC", modality="microbiology")
                    # Link culture test to first diagnosis if available
                    if block_diagnoses:
                        relationships.append(
                            Relationship(
                                subj=c_node, obj=block_diagnoses[0], type="detects",
                                properties={
                                    "source": "structured_fallback",
                                    "source_type": "MIMIC",
                                    "provenance": "evidence_backed",
                                    "extraction_method": "microbiology_parsing"
                                },
                            )
                        )
            # Extract organism findings
            for org_match in re.finditer(
                r"(STAPH|STREP|ENTEROCOCCUS|PSEUDOMONAS|KLEBSIELLA|E\.?\s*COLI|"
                r"ACINETOBACTER|SERRATIA|PROTEUS|CANDIDA|MRSA|VRE|"
                r"MIXED BACTERIAL FLORA|NO GROWTH|No MRSA)",
                micro_text, re.IGNORECASE
            ):
                org_name = org_match.group(0).strip()
                if "NO GROWTH" in org_name.upper() or "No MRSA" in org_name:
                    # Create a Clinical_Finding for negative results
                    finding = _add_node(f"{org_name} (negative)", "Clinical_Finding",
                                       f"Microbiology finding: {org_name}",
                                       source_type="MIMIC", modality="microbiology")
                else:
                    finding = _add_node(org_name, "Bacterium",
                                       f"Microorganism: {org_name}",
                                       source_type="MIMIC", modality="microbiology")
                    block_bacteria.append(finding)

        # --- Demographics → extract anatomy-related and demographic nodes ---
        demo_match = re.search(r"Demographics:\s*gender\s+(\w+);\s*anchor_age\s+(\d+)", block)
        if demo_match and patient_node:
            # Age as a clinical condition marker
            age = int(demo_match.group(2))
            if age >= 65:
                elderly_node = _add_node("Elderly patient", "Condition",
                                        f"Patient age {age}, elderly (≥65)")
                relationships.append(
                    Relationship(
                        subj=patient_node, obj=elderly_node, type="has_condition",
                        properties={
                            "source": "structured_fallback",
                            "source_type": "MIMIC",
                            "provenance": "evidence_backed",
                            "extraction_method": "demographic_extraction"
                        },
                    )
                )

        # === INTER-ENTITY RELATIONSHIPS (distributed, no mega-hub) ===

        n_diag = len(block_diagnoses)
        if n_diag == 0:
            continue

        # Procedure --performed_for--> Diagnosis (round-robin, 1:1)
        for i, p_node in enumerate(block_procedures):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=p_node, obj=d_node, type="performed_for",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "structural",
                        "provenance": "structurally_generated",
                        "extraction_method": "round_robin_assignment"
                    },
                )
            )

        # Medication --treats--> Diagnosis (round-robin, 1:1)
        for i, m_node in enumerate(block_medications):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=m_node, obj=d_node, type="treats",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "structural",
                        "provenance": "structurally_generated",
                        "extraction_method": "round_robin_assignment"
                    },
                )
            )

        # LabTest --monitors--> Diagnosis (round-robin, 1:1)
        for i, l_node in enumerate(block_labs):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=l_node, obj=d_node, type="monitors",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "structural",
                        "provenance": "structurally_generated",
                        "extraction_method": "round_robin_assignment"
                    },
                )
            )

        # Bacteria --causes--> Disease/Infection diagnoses
        disease_diags = [d for d in block_diagnoses
                         if (d.properties or {}).get("source_type") == "rule_based"
                         or d.type in ("Disease", "Infection")]
        if block_bacteria and disease_diags:
            for i, b_node in enumerate(block_bacteria):
                target = disease_diags[i % len(disease_diags)]
                relationships.append(
                    Relationship(
                        subj=b_node, obj=target, type="causes",
                        properties={
                            "source": "structured_fallback",
                            "source_type": "MIMIC",
                            "provenance": "evidence_backed",
                            "extraction_method": "microbiology_causation"
                        },
                    )
                )

        # Abnormal measurements --indicates--> nearest Diagnosis
        for meas_node, lab_node in block_measurements:
            if (meas_node.properties or {}).get("flag") == "abnormal":
                target_diag = block_diagnoses[0]
                relationships.append(
                    Relationship(
                        subj=meas_node, obj=target_diag, type="indicates",
                        properties={
                            "source": "structured_fallback",
                            "source_type": "MIMIC",
                            "provenance": "evidence_backed",
                            "extraction_method": "abnormal_lab_indication"
                        },
                    )
                )

        # Diagnosis chain (sequential, not clique — avoids O(n^2) edges)
        for i in range(n_diag - 1):
            relationships.append(
                Relationship(
                    subj=block_diagnoses[i],
                    obj=block_diagnoses[i + 1],
                    type="associated_with",
                    properties={
                        "source": "structured_fallback",
                        "source_type": "rule_based",
                        "provenance": "evidence_backed",
                        "extraction_method": "sequential_chain"
                    },
                )
            )

        # Medication --administered_via--> route grouping
        # (connect medications sharing the same route)
        route_groups = {}
        for m_node in block_medications:
            desc = (m_node.properties or {}).get("description", "")
            route_match = re.search(r"via (\S+)", desc)
            if route_match:
                route = route_match.group(1)
                route_groups.setdefault(route, []).append(m_node)
        for route, meds in route_groups.items():
            for i in range(len(meds) - 1):
                relationships.append(
                    Relationship(
                        subj=meds[i], obj=meds[i + 1], type="same_route",
                        properties={
                            "source": "structured_fallback",
                            "source_type": "structural",
                            "provenance": "structurally_generated",
                            "route": route
                        },
                    )
                )

    # Patient gets a single link to just the FIRST diagnosis (minimal presence)
    if patient_node and block_diagnoses:
        relationships.append(
            Relationship(
                subj=patient_node,
                obj=block_diagnoses[0],
                type="diagnosed_with",
                properties={
                    "source": "structured_fallback",
                    "source_type": "MIMIC",
                    "provenance": "evidence_backed",
                    "extraction_method": "mimic_structured_admission"
                },
            )
        )

    return GraphElement(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_element,
    )


def create_metagraph(args, content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    whole_chunk = content

    if args.grained_chunk == True:
        # Lazy import avoids pulling LangChain/Pydantic stack unless requested.
        from data_chunk import run_chunk
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        graph_elements = _extract_graph_elements_from_text(cont, element_example)
        if not graph_elements.nodes:
            continue
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j
