"""
Unit tests for Graph Provenance (Phase 2).
Verifies that all nodes and relationships retain verifiable provenance metadata.
"""

import pytest
from create_graph import _fallback_extract_graph_elements
from camel.loaders.unstructured_io import UnstructuredIO


def test_provenance_patient_clinical_data():
    sample_text = """
patient_id: 10000690
history of present illness:
Demographics: gender F; anchor_age 86; date_of_death 2152-01-30.
Admission summary: hadm_id 23280645; admit 2150-09-16 19:48:00; discharge 2150-09-24 13:50:00; type EW EMER.; from EMERGENCY ROOM; to SKILLED NURSING FACILITY; race WHITE; insurance Medicare; hospital_expire_flag 0.
Diagnoses: [1] Congestive heart failure, unspecified; [2] Pneumonia, organism unspecified.
Procedures: Insertion of endotracheal tube on 2150-09-17.
Medications: CeftriaXONE 1 gm via IV; Vancomycin 1000 mg via IV.
Recent labs: White Blood Cells: 14.8 K/uL (flag abnormal); Creatinine: 1.2 mg/dL.
clinical impression: Synthetic note.
"""
    uio = UnstructuredIO()
    source_element = uio.create_element_from_text(text=sample_text)
    graph_element = _fallback_extract_graph_elements(sample_text, source_element)

    assert len(graph_element.nodes) > 0
    assert len(graph_element.relationships) > 0

    # Verify Node Provenance
    for node in graph_element.nodes:
        props = node.properties
        assert "source" in props
        assert "source_type" in props
        assert "modality" in props
        assert "extraction_confidence" in props
        assert "extraction_method" in props
        assert props["source"] == "structured_fallback"

    # Verify Relationship Provenance
    for rel in graph_element.relationships:
        props = rel.properties
        assert "source" in props
        assert "source_type" in props
        assert "provenance" in props
        # Provenance must be explicitly distinguished (e.g. evidence_backed vs structurally_generated)
        assert props["provenance"] in ["evidence_backed", "structurally_generated", "dictionary_derived"]


def test_provenance_dictionary_data():
    sample_dict_text = """
medical dictionary entries:
DIAGNOSIS code=486 icd_version=9 name=Pneumonia, organism unspecified
PROCEDURE code=9604 icd_version=9 name=Insertion of endotracheal tube
LAB_TEST itemid=51221 label=Hematocrit fluid=Blood category=Hematology
"""
    uio = UnstructuredIO()
    source_element = uio.create_element_from_text(text=sample_dict_text)
    graph_element = _fallback_extract_graph_elements(sample_dict_text, source_element)

    assert len(graph_element.nodes) >= 3
    for node in graph_element.nodes:
        assert node.properties.get("source_type") in ["dictionary", "rule_based"]
        assert node.properties.get("modality") in ["dictionary", "structured_data"]
