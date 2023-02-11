DROP TABLE IF EXISTS hep_miniicustays;
CREATE TABLE hep_miniicustays AS (SELECT * FROM mimiciv_icu.icustays WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimiciv_icu.inputevents WHERE itemid IN (225152)));
DROP TABLE IF EXISTS hep_miniinputevents;
CREATE TABLE hep_miniinputevents AS (SELECT * FROM mimiciv_icu.inputevents WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimiciv_icu.inputevents WHERE itemid IN (225152)));

DROP TABLE IF EXISTS heparin_diagnoses CASCADE;
CREATE TABLE heparin_diagnoses AS
WITH hep_times AS
(
	SELECT hin.stay_id
	, min(hin.starttime) - interval '1' day AS starttime
	, max (hin.endtime) + interval '1' day AS endtime
	FROM hep_miniinputevents hin
	WHERE itemid IN (225152)
	GROUP BY hin.stay_id
)
, timewinds AS
(
	SELECT ht.stay_id
	, icust.hadm_id
	, CASE
		WHEN ht.starttime <= icust.intime AND icust.intime IS NOT NULL THEN icust.intime
	ELSE ht.starttime END AS starttime
	, CASE
		WHEN ht.endtime >= icust.outtime AND icust.outtime IS NOT NULL THEN icust.outtime
	ELSE ht.endtime END AS endtime
	, adm.admittime
	, adm.dischtime
	FROM hep_times ht
	LEFT JOIN hep_miniicustays icust
	ON ht.stay_id = icust.stay_id
	LEFT JOIN mimiciv_hosp.admissions adm
	ON adm.hadm_id = icust.hadm_id
)
-- maybe add storeime
, interdiagnoses AS
(
	SELECT tmw.stay_id
		, dd.seq_num
		, dd.icd_code
		, dd.icd_version
	FROM timewinds tmw
	LEFT JOIN mimiciv_hosp.diagnoses_icd dd
		ON dd.hadm_id = tmw.hadm_id
		AND tmw.admittime <= tmw.starttime
	WHERE icd_code IN ('4019','4280','42731','41401','41071','V4582','78551','412','42823','2449','53081','V5861','2859','2851','99592''389','78552','2724','25000','V5867','27800','27801','311','51881','486','4168','496','41519','32723','5849','2762','40390','5990','5859','2761','5845','V1582','3051','I2510','I10','I4891','I214','I480','R570','I130','I110','I5023','E039','K219','Z7901','D62','Z7902','D649','D696','R6521','A419','E785','Z794','E1122','E119','E669','F329','F419','J9601','J189','J449','G4733','N179','E872','E871','N170','N390','E875','N189','Z87891','F17210')
)
SELECT id.stay_id
	, CASE
		WHEN id.icd_version = 9 THEN
			CASE
				WHEN id.icd_code in ('4019','4280','42731','41401','41071','V4582','78551','412','42823') THEN 'CVS'
				WHEN id.icd_code in ('2449') THEN 'Endocrine'
				WHEN id.icd_code in ('53081') THEN 'GI'
				WHEN id.icd_code in ('V5861','2859','2851') THEN 'Hematological'
				WHEN id.icd_code in ('99592','389','78552') THEN 'Infectious'
				WHEN id.icd_code in ('2724','25000','V5867') THEN 'Met'
				WHEN id.icd_code in ('27800','27801') THEN 'Obes'
				WHEN id.icd_code in ('311') THEN 'Psych'
				WHEN id.icd_code in ('51881','486','4168','496','41519','32723') THEN 'Pulmonary'
				WHEN id.icd_code in ('5849','2762','40390','5990','5859','2761','5845') THEN 'Renal'
				WHEN id.icd_code in ('V1582','3051') THEN 'Smoking'
			ELSE NULL END
	ELSE 
		CASE
			WHEN id.icd_version = 10 THEN
				CASE
					WHEN id.icd_code in ('I2510','I10','I4891','I214','I480','R570','I130','I110','I5023') THEN 'CVS'
					WHEN id.icd_code in ('E039') THEN 'Endocrine'
					WHEN id.icd_code in ('K219') THEN 'GI'
					WHEN id.icd_code in ('Z7901','D62','Z7902','D649','D696') THEN 'Hematological'
					WHEN id.icd_code in ('R6521','A419') THEN 'Infectious'
					WHEN id.icd_code in ('E785','Z794','E1122','E119') THEN 'Met'
					WHEN id.icd_code in ('E669') THEN 'Obes'
					WHEN id.icd_code in ('F329','F419') THEN 'Psych'
					WHEN id.icd_code in ('J9601','J189','J449','G4733') THEN 'Pulmonary'
					WHEN id.icd_code in ('N179','E872','E871','N170','N390','E875','N189') THEN 'Renal'
					WHEN id.icd_code in ('Z87891','F17210') THEN 'Smoking'
				ELSE NULL END
		ELSE NULL END 
		END AS category
	, id.icd_code
	, id.icd_version
FROM timewinds tmw
LEFT JOIN interdiagnoses id
	ON id.stay_id = tmw.stay_id;

	
DROP TABLE IF EXISTS hep_miniicustays;
DROP TABLE IF EXISTS hep_miniinputevents;
