
\i build_VD_chartevents.sql
\copy vancomycin_chartevents to '../VD_chartevents.csv' csv header
DROP TABLE vancomycin_chartevents CASCADE;
------------------------------------------------------------------
\i build_VD_labevents.sql
\copy vancomycin_labevents to '../VD_labevents.csv' csv header
DROP TABLE vancomycin_labevents CASCADE;
------------------------------------------------------------------
\i build_VD_diagnoses.sql
\copy vancomycin_diagnoses to '../VD_diagnoses.csv' csv header
DROP TABLE vancomycin_diagnoses CASCADE;



