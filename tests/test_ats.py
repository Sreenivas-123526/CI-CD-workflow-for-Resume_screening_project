import os
import sys

# Add the project root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ats import calculate_ats_score


def test_calculate_ats_score():
    resume = "Python SQL Machine Learning"
    jd = "Python SQL"

    score = calculate_ats_score(resume, jd)

    assert score >= 0
