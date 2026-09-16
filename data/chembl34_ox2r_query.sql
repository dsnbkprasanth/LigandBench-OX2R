-- Reproducibility query specification for ChEMBL 34 / human OX2R CHEMBL4792.
-- Execute against the ChEMBL 34 SQLite/MySQL/PostgreSQL release.
-- The frozen curated dataset shipped with this repository is preferred for
-- exact reruns of the published benchmark.

SELECT
    md.chembl_id AS molecule_chembl_id,
    cs.canonical_smiles,
    td.chembl_id AS target_chembl_id,
    a.assay_chembl_id,
    a.assay_type,
    a.description AS assay_description,
    doc.chembl_id AS document_chembl_id,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.action_type
FROM target_dictionary td
JOIN assays a ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
JOIN molecule_dictionary md ON act.molregno = md.molregno
LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
LEFT JOIN docs doc ON a.doc_id = doc.doc_id
WHERE td.chembl_id = 'CHEMBL4792';
