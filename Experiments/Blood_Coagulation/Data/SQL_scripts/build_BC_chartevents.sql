DROP TABLE IF EXISTS hep_miniicustays;
CREATE TABLE hep_miniicustays AS (SELECT * FROM mimic_icu.icustays WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimic_icu.inputevents WHERE itemid IN (225152)));
-- DROP TABLE IF EXISTS hep_minichartevents;
-- CREATE TABLE hep_minichartevents AS (SELECT * FROM mimic_icu.chartevents WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimic_icu.inputevents WHERE itemid IN (225152)));
DROP TABLE IF EXISTS hep_miniinputevents;
CREATE TABLE hep_miniinputevents AS (SELECT * FROM mimic_icu.inputevents WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimic_icu.inputevents WHERE itemid IN (225152)));

DROP TABLE IF EXISTS heparin_chartevents CASCADE;
CREATE TABLE heparin_chartevents AS
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
	, CASE
		WHEN ht.starttime <= icust.intime AND icust.intime IS NOT NULL THEN icust.intime
	ELSE ht.starttime END AS starttime
	, CASE
		WHEN ht.endtime >= icust.outtime AND icust.outtime IS NOT NULL THEN icust.outtime
	ELSE ht.endtime END AS endtime 
	FROM hep_times ht
	LEFT JOIN hep_miniicustays icust
	ON ht.stay_id = icust.stay_id
)
, interchartevents AS
(
	SELECT mc.stay_id
		, mc.itemid
		, mc.charttime
		, mc.value
		, mc.valuenum
		, mc.valueuom
	FROM timewinds tmw
	LEFT JOIN mimic_icu.chartevents mc
		ON mc.stay_id = tmw.stay_id
		AND mc.charttime >= tmw.starttime
		AND mc.charttime <= tmw.endtime
	WHERE itemid IN (225958,224145,225152,229373,229375,227466,220235,220045,220562,227456,220574,220051,220050,225651,225690,220615,229761,223900,220739,223901,220545,226540,220228,227467,220561,223830,225678,227457,227465,220560,220210,220227,227428,223762,223761,227429,220546,227468,220541,220612,227444,220507,226512,226531,220640,227442,226535,227464,225667,225625,228390,226534,228389,220645,224639,225636)
)
, interinputevents AS
(
	SELECT mi.stay_id
		, mi.itemid
		, mi.starttime
		, mi.endtime
		, mi.amount
		, mi.amountuom
		, mi.patientweight
	FROM timewinds tmw
	LEFT JOIN hep_miniinputevents mi
		ON mi.stay_id = tmw.stay_id
		AND mi.starttime >= tmw.starttime
		AND mi.endtime <= tmw.endtime
		AND mi.starttime <= mi.endtime
	WHERE itemid IN (225958,224145,225152,229373,229375,227466,220235,220045,220562,227456,220574,220051,220050,225651,225690,220615,229761,223900,220739,223901,220545,226540,220228,227467,220561,223830,225678,227457,227465,220560,220210,220227,227428,223762,223761,227429,220546,227468,220541,220612,227444,220507,226512,226531,220640,227442,226535,227464,225667,225625,228390,226534,228389,220645,224639,225636)
)
, heparin_data AS
(
	SELECT ic.stay_id
		, ic.itemid
		, extract(epoch from ic.charttime) - extract(epoch from tmw.starttime) AS charttime
		, ic.value
		, ic.valuenum
		, ic.valueuom
		, null AS starttime
		, null AS endtime
		, null AS amount
		, null AS amountuom
		, null AS patientweight
	FROM timewinds tmw
	LEFT JOIN interchartevents ic
		ON ic.stay_id = tmw.stay_id
	UNION
	SELECT ii.stay_id
		, ii.itemid
		, null AS charttime
		, null AS value
		, null AS valuenum
		, null AS valueuom
		, extract(epoch from ii.starttime) - extract(epoch from tmw.starttime) AS starttime
		, extract(epoch from ii.endtime) - extract(epoch from tmw.starttime) AS endtime
		, ii.amount
		, ii.amountuom
		, ii.patientweight
	FROM timewinds tmw
	LEFT JOIN interinputevents ii
		ON ii.stay_id = tmw.stay_id
)
SELECT hd.*
	, adm.ethnicity
	, adm.admission_type
	, adm.hospital_expire_flag
	, pt.gender
	, extract(year from tmw.starttime) - (pt.anchor_year - pt.anchor_age) AS age
FROM hep_miniicustays mi
LEFT JOIN heparin_data hd
	ON hd.stay_id= mi.stay_id
LEFT JOIN mimic_core.admissions adm
	ON adm.hadm_id = mi.hadm_id
LEFT JOIN mimic_core.patients pt
	ON pt.subject_id = adm.subject_id
LEFT JOIN timewinds tmw
	ON tmw.stay_id = mi.stay_id;
	
	
DROP TABLE IF EXISTS hep_miniicustays;
-- DROP TABLE IF EXISTS hep_minichartevents;
DROP TABLE IF EXISTS hep_miniinputevents;
