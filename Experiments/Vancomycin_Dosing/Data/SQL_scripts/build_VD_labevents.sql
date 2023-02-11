DROP TABLE IF EXISTS vanco_miniicustays;
CREATE TABLE vanco_miniicustays AS (SELECT * FROM mimiciv_icu.icustays WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimiciv_icu.inputevents WHERE itemid IN (225798)));
DROP TABLE IF EXISTS vanco_miniinputevents;
CREATE TABLE vanco_miniinputevents AS (SELECT * FROM mimiciv_icu.inputevents WHERE stay_id IN (SELECT DISTINCT(stay_id) FROM mimiciv_icu.inputevents WHERE itemid IN (225798)));

DROP TABLE IF EXISTS vancomycin_labevents CASCADE;
CREATE TABLE vancomycin_labevents AS
WITH vanco_times AS
(
	SELECT hin.stay_id
	, min(hin.starttime) - interval '1' day AS starttime
	, max (hin.endtime) + interval '1' day AS endtime
	FROM vanco_miniinputevents hin
	WHERE itemid IN (225798)
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
	FROM vanco_times ht
	LEFT JOIN vanco_miniicustays icust
	ON ht.stay_id = icust.stay_id
)
-- maybe add storeime
, interlabevents AS
(
	SELECT tmw.stay_id
		, ml.itemid
		, ml.charttime
		, ml.storetime
		, ml.value
		, ml.valuenum
		, ml.valueuom
	FROM timewinds tmw
	LEFT JOIN mimiciv_hosp.labevents ml
		ON ml.hadm_id = tmw.hadm_id
		AND ml.charttime >= tmw.starttime
		AND ml.charttime <= tmw.endtime
	WHERE itemid IN (51009,50861,50878,51300,51274,51275,50889,50912,51006,50971,50893,50983,52618,50883,50885,51221,51222,51237,51265,51003,52111,51214,51196,50915)
)
SELECT il.stay_id
	, il.itemid
	, extract(epoch from il.charttime) - extract(epoch from tmw.starttime) AS charttime
	, extract(epoch from il.storetime) - extract(epoch from tmw.starttime) AS storetime
	, il.value
	, il.valuenum
	, il.valueuom
FROM timewinds tmw
LEFT JOIN interlabevents il
	ON il.stay_id = tmw.stay_id;

	
	
DROP TABLE IF EXISTS vanco_miniicustays;
DROP TABLE IF EXISTS vanco_miniinputevents;
