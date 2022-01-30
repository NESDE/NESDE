
\i build_BC_chartevents.sql
\copy heparin_chartevents to '../BC_chartevents.csv' csv header
DROP TABLE heparin_chartevents CASCADE;
------------------------------------------------------------------
\i build_BC_labevents.sql
\copy heparin_labevents to '../BC_labevents.csv' csv header
DROP TABLE heparin_labevents CASCADE;
------------------------------------------------------------------
\i build_BC_diagnoses.sql
\copy heparin_diagnoses to '../BC_diagnoses.csv' csv header
DROP TABLE heparin_diagnoses CASCADE;



