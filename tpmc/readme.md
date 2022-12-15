This directory runs a Test Particle Monte Carlo (TPMC) model through a geometry defined by a STL file.

To run execute: run_tpmc.py

File Discriptions:
-- run_tpmc.py: defines flow conditions, solution parameters, and geometry. Creates a CASE_TPMC object.

-- case_tpmc.py: defines a CASE_TPMC class. This contains the execute_case() method that is called to run the TPMC simulation.

-- postprocess.py: class for calculating outputs and generating plots.

-- tests.py: a few tests to verify that functions behave as expected. Not used in normal simulation.

-- utilitis.py: functions that are handy to have. I think most of these could have been made class methods in case_tpmc.py, but it works regardless.

-- *.stl: stl files that define the shape of the inlet, walls, and outlet.
