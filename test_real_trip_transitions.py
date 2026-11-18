
import math
import pandas as pd

from src.data.loader import load_trip
from src.features.pipeline import process_trip
from src.decoder.transitions import compute_transitions


TRIP_ID = "00000001"


def normalize_timestamps(dataframe, column="timestamp"):
    """
    Convert timestamps to pandas datetime values.

    Supports:
    - Existing datetime values
    - Datetime strings
    - Numeric Unix timestamps in seconds, milliseconds,
      microseconds, or nanoseconds
    - Numeric elapsed-time values in seconds
    """

    if column not in dataframe.columns:
        raise ValueError(f"Missing timestamp column: {column}")

    values = dataframe[column]

    if pd.api.types.is_datetime64_any_dtype(values):
        dataframe[column] = pd.to_datetime(values)
        return dataframe

    numeric_values = pd.to_numeric(values, errors="coerce")

    # Handle numeric timestamps.
    if numeric_values.notna().all():

        if numeric_values.empty:
            raise ValueError("Timestamp column is empty.")

        magnitude = numeric_values.abs().median()

        if magnitude >= 1e17:
            unit = "ns"

        elif magnitude >= 1e14:
            unit = "us"

        elif magnitude >= 1e11:
            unit = "ms"

        else:
            unit = "s"

        print(f"Converting numeric timestamps using unit: {unit}")

        dataframe[column] = pd.to_datetime(
            numeric_values,
            unit=unit,
            origin="unix",
            errors="raise"
        )

    else:
        # Handle datetime strings.
        dataframe[column] = pd.to_datetime(
            values,
            errors="raise"
        )

    if dataframe[column].isna().any():
        raise ValueError("Timestamp column contains missing values.")

    return dataframe


def main():

    print("\n========== REAL TRIP TRANSITION TEST ==========")

    # --------------------------------------------------
    # 1. Load the actual trip
    # --------------------------------------------------

    print("\n[1] Loading trip...")

    trip = load_trip(
        trip_id=TRIP_ID,
        split="train",
        data_dir="data"
    )

    track = trip["track"].copy()
    track["trip_id"] = TRIP_ID

    print("\nTimestamp diagnostics:")

    print("Timestamp dtype:", track["timestamp"].dtype)

    print("First 10 timestamps:")
    print(track["timestamp"].head(10).tolist())

    # Convert observation timestamps.
    track = normalize_timestamps(track)

    observations = track[
        [
            "trip_id",
            "obs_idx",
            "timestamp",
            "longitude",
            "latitude"
        ]
    ].copy()

    print("\nTrip loaded:", TRIP_ID)
    print("GPS observations:", len(observations))

    print("Normalized observation timestamps:")
    print(observations["timestamp"].head(5).tolist())

    # --------------------------------------------------
    # 2. Build road graph and generate candidates
    # --------------------------------------------------

    print("\n[2] Building road graph and generating candidates...")

    result = process_trip(
        trip_id=TRIP_ID,
        split="train",
        data_dir="data",
        radius_m=50,
        max_candidates=8,
        save=False
    )

    road_graph = result["road_graph"]
    candidates = result["candidates"].copy()

    # IMPORTANT:
    # process_trip loads the trip independently.
    # Therefore, normalize its candidate timestamps too.
    candidates = normalize_timestamps(candidates)

    print("Road graph type:", type(road_graph))
    print("Graph directed:", road_graph.is_directed())
    print("Graph is multigraph:", road_graph.is_multigraph())
    print("Graph nodes:", road_graph.number_of_nodes())
    print("Graph edges:", road_graph.number_of_edges())
    print("Candidate rows:", len(candidates))

    print("\nCandidate timestamp dtype:", candidates["timestamp"].dtype)
    print("First 5 candidate timestamps:")
    print(candidates["timestamp"].head(5).tolist())

    # --------------------------------------------------
    # 3. Verify graph edge attributes
    # --------------------------------------------------

    print("\n[3] Checking graph edge attributes...")

    required_attributes = {
        "arc_id",
        "from_node_id",
        "to_node_id",
        "length_m"
    }

    missing_edges = []

    for u, v, key, data in road_graph.edges(
        keys=True,
        data=True
    ):

        missing = required_attributes - set(data.keys())

        if missing:
            missing_edges.append(
                (u, v, key, missing)
            )

    if missing_edges:

        print("Edges with missing attributes:", len(missing_edges))

        for edge in missing_edges[:10]:
            print(edge)

        raise ValueError(
            "Some graph edges have missing attributes."
        )

    print("All graph edges contain required attributes.")

    # --------------------------------------------------
    # 4. Identify zero-length road arcs
    # --------------------------------------------------

    print("\n[4] Checking zero-length road arcs...")

    zero_edges = []

    for u, v, key, data in road_graph.edges(
        keys=True,
        data=True
    ):

        length = float(data["length_m"])

        if not math.isfinite(length) or length < 0:

            zero_edges.append(
                (u, v, key, data)
            )

    zero_arc_ids = {
        data["arc_id"]
        for _, _, _, data in zero_edges
    }

    print("Invalid-length edge count:", len(zero_edges))

    if zero_edges:

        print("\nProblematic road arcs:")

        for u, v, key, data in zero_edges[:10]:

            print(
                "Arc ID:", data["arc_id"],
                "| From node:", u,
                "| To node:", v,
                "| Length:", data["length_m"]
            )

    else:
        print("No negative or non-finite road lengths found.")

    # --------------------------------------------------
    # 5. Check whether zero-length arcs are candidates
    # --------------------------------------------------

    print("\n[5] Checking zero-length candidate arcs...")

    zero_length_arc_ids = {
        data["arc_id"]
        for _, _, _, data in road_graph.edges(
            keys=True,
            data=True
        )
        if float(data["length_m"]) == 0.0
    }

    candidate_arc_ids = set(
        candidates["candidate_arc_id"]
    )

    affected_arc_ids = (
        zero_length_arc_ids.intersection(candidate_arc_ids)
    )

    print("Zero-length graph arcs:", len(zero_length_arc_ids))

    print(
        "Zero-length arcs appearing in candidates:",
        len(affected_arc_ids)
    )

    if affected_arc_ids:

        print(
            "Affected candidate arc IDs:",
            sorted(affected_arc_ids)
        )

        affected_rows = candidates[
            candidates["candidate_arc_id"].isin(
                affected_arc_ids
            )
        ]

        print(
            "Affected candidate rows:",
            len(affected_rows)
        )

        print("\nAffected candidate details:")

        print(
            affected_rows.head(10).to_string(index=False)
        )

    else:

        print(
            "No zero-length arcs are used as GPS candidates."
        )

    # --------------------------------------------------
    # 6. Inspect the specific failing arc
    # --------------------------------------------------

    print("\n[6] Inspecting arc 51615...")

    found = False

    for u, v, key, data in road_graph.edges(
        keys=True,
        data=True
    ):

        if data["arc_id"] == 51615:

            found = True

            print("Arc ID:", data["arc_id"])
            print("From node:", u)
            print("To node:", v)
            print("Length:", data["length_m"])
            print("Geometry:", data.get("geometry"))

            print("\nFrom-node attributes:")
            print(road_graph.nodes[u])

            print("\nTo-node attributes:")
            print(road_graph.nodes[v])

            break

    if not found:
        print("Arc 51615 was not found.")

    # --------------------------------------------------
    # 7. Validate timestamps before transition calculation
    # --------------------------------------------------

    print("\n[7] Validating timestamps...")

    print(
        "Observation timestamp dtype:",
        observations["timestamp"].dtype
    )

    print(
        "Candidate timestamp dtype:",
        candidates["timestamp"].dtype
    )

    if not pd.api.types.is_datetime64_any_dtype(
        observations["timestamp"]
    ):
        raise TypeError(
            "Observation timestamps are not datetime values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        candidates["timestamp"]
    ):
        raise TypeError(
            "Candidate timestamps are not datetime values."
        )

    print("Observation timestamps:", len(observations))
    print("Candidate timestamps:", len(candidates))
    print("Timestamp validation successful.")

    # --------------------------------------------------
    # 8. Calculate transition scores
    # --------------------------------------------------

    print("\n[8] Computing real transition scores...")

    try:

        transitions = compute_transitions(
            candidates=candidates,
            road_graph=road_graph,
            observations=observations,
            beta_m=50.0,
            max_speed_mps=55.0,
            speed_scale_mps=10.0
        )

    except (ValueError, TypeError) as error:

        print("\nTRANSITION CALCULATION FAILED")

        print("Error type:", type(error).__name__)
        print("Error message:", error)

        print(
            "\nNo graph edges or candidate rows were modified."
        )

        return

    # --------------------------------------------------
    # 9. Inspect transition results
    # --------------------------------------------------

    print("\n[9] Transition results...")

    trip_transitions = transitions[TRIP_ID]

    total_observations = len(trip_transitions)

    observations_with_transitions = sum(
        bool(pairs)
        for pairs in trip_transitions.values()
    )

    total_pairs = sum(
        len(pairs)
        for pairs in trip_transitions.values()
    )

    finite_score_pairs = sum(
        math.isfinite(score.log_score)
        for pairs in trip_transitions.values()
        for score in pairs.values()
    )

    print(
        "Observations represented:",
        total_observations
    )

    print(
        "Observations with transitions:",
        observations_with_transitions
    )

    print(
        "Total candidate transition pairs:",
        total_pairs
    )

    print(
        "Finite-score transition pairs:",
        finite_score_pairs
    )

    print(
        "Non-finite-score pairs:",
        total_pairs - finite_score_pairs
    )

    # --------------------------------------------------
    # 10. Display sample transition scores
    # --------------------------------------------------

    print("\n[10] Sample transition scores...")

    displayed = 0

    for obs_idx, pairs in trip_transitions.items():

        for (previous_arc, current_arc), score in pairs.items():

            print(
                f"Observation {obs_idx}: "
                f"{previous_arc} -> {current_arc} | "
                f"Network distance: {score.network_distance_m} | "
                f"Log score: {score.log_score}"
            )

            displayed += 1

            if displayed >= 5:
                break

        if displayed >= 5:
            break

    print(
        "\n========== TEST COMPLETED SUCCESSFULLY =========="
    )


if __name__ == "__main__":
    main()
