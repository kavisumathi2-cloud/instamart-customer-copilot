"""The supplied generate_data.py omits expected_escalated_amount_inr."""

import pandas as pd

import src.refund_agent as agent


def test_missing_escalation_column_is_derived():
    legacy = pd.DataFrame(
        {
            "scenario_id": ["S1", "S3", "S4"],
            "expected_outcome": ["REFUND", "ESCALATE", "DENIED"],
            "expected_refund_inr": [128.25, 1999.0, 0.0],
        }
    )

    result = agent.normalize_scenarios(legacy)

    assert result["expected_escalated_amount_inr"].tolist() == [
        0.0,
        1999.0,
        0.0,
    ]
    assert result["expected_refund_inr"].tolist() == [128.25, 0.0, 0.0]


def test_existing_escalation_column_is_unchanged():
    current = pd.DataFrame(
        {
            "scenario_id": ["S3"],
            "expected_outcome": ["ESCALATE"],
            "expected_refund_inr": [0.0],
            "expected_escalated_amount_inr": [1999.0],
        }
    )

    result = agent.normalize_scenarios(current)

    assert result["expected_escalated_amount_inr"].tolist() == [1999.0]
    assert result["expected_refund_inr"].tolist() == [0.0]
