import datetime as dt
from pathlib import Path

import altair as alt
import fastf1
import numpy as np
import pandas as pd
import streamlit as st


def enable_fastf1_cache():
    cache_dir = Path("./fastf1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


@st.cache_data(show_spinner=False)
def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Ensure we have a clean, small set of columns for the UI
    df = schedule[["RoundNumber", "EventName", "EventDate", "Country"]].copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"]).dt.date
    return df


@st.cache_data(show_spinner=True)
def load_race_results(year: int, round_number: int) -> pd.DataFrame:
    session = fastf1.get_session(year, round_number, "R")
    session.load()  # uses cache after first run
    results = session.results
    # Convert to DataFrame with the fields we need
    df = results[["DriverNumber", "Abbreviation", "TeamName", "Position"]].copy()
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    return df


def build_prediction_for_event(
    year: int,
    round_number: int,
    history_races: int,
    seasons_back: int = 2,
) -> pd.DataFrame:
    """
    Build predictions using up to `history_races` most recent completed races,
    going backwards from the selected race across this season and previous ones.
    Only the current grid (from the most recent race in that window) is included,
    so you should see each active driver once.
    """
    history_frames: list[tuple[int, int, pd.DataFrame]] = []
    races_collected = 0

    # Walk backwards over seasons and rounds until we have enough races
    for y in range(year, max(year - seasons_back, 2014) - 1, -1):
        # Figure out which rounds in this season to consider (backwards)
        if y == year:
            start_round = round_number - 1
            if start_round < 1:
                continue
            candidate_rounds = range(start_round, 0, -1)
        else:
            sched = get_event_schedule(y)
            if sched.empty:
                continue
            max_round = int(sched["RoundNumber"].max())
            candidate_rounds = range(max_round, 0, -1)

        for r in candidate_rounds:
            if races_collected >= history_races:
                break
            try:
                df_r = load_race_results(y, r)
                if df_r.empty:
                    continue
                df_r["Round"] = r
                df_r["Year"] = y
                history_frames.append((y, r, df_r))
                races_collected += 1
            except Exception:
                # If a particular session fails to load, skip it
                continue

        if races_collected >= history_races:
            break

    if not history_frames:
        return pd.DataFrame(columns=["Driver", "Team", "AvgFinish", "RacesUsed"])

    # Combine all races into one table
    history = pd.concat([df for _, _, df in history_frames], ignore_index=True)
    history = history.dropna(subset=["Position"])

    if history.empty:
        return pd.DataFrame(columns=["Driver", "Team", "AvgFinish", "RacesUsed"])

    # Restrict to the current grid: take drivers from the most recent race in our window
    latest_year, latest_round = max((y, r) for (y, r, _) in history_frames)
    latest_df = next(df for (y, r, df) in history_frames if y == latest_year and r == latest_round)
    current_grid = set(latest_df["Abbreviation"].dropna().unique())

    history = history[history["Abbreviation"].isin(current_grid)]

    if history.empty:
        return pd.DataFrame(columns=["Driver", "Team", "AvgFinish", "RacesUsed"])

    # Get latest metadata (team and number) per driver from the most recent race where they appeared
    history_sorted = history.sort_values(["Year", "Round"])
    latest_meta = (
        history_sorted.groupby("Abbreviation")[["DriverNumber", "TeamName"]]
        .last()
        .reset_index()
    )

    # Aggregate performance stats per driver
    stats = (
        history.groupby("Abbreviation")
        .agg(
            AvgFinish=("Position", "mean"),
            RacesUsed=("Position", "count"),
        )
        .reset_index()
    )

    summary = stats.merge(latest_meta, on="Abbreviation", how="left")
    summary = summary.sort_values("AvgFinish", ascending=True).reset_index(drop=True)
    summary.insert(0, "PredictedRank", np.arange(1, len(summary) + 1))
    summary.rename(
        columns={"Abbreviation": "Driver", "TeamName": "Team"},
        inplace=True,
    )
    return summary


def main():
    st.set_page_config(
        page_title="F1 Race Predictor",
        page_icon="🏎️",
        layout="wide",
    )

    st.title("F1 Race Prediction Dashboard")
    st.markdown(
        "Uses **FastF1** data to stay up-to-date with the current and upcoming races. "
        "Predictions are based on recent race finishing positions across this and recent seasons."
    )

    enable_fastf1_cache()

    # Sidebar controls
    today = dt.date.today()
    current_year = today.year

    with st.sidebar:
        st.header("Configuration")
        year = st.number_input("Season year", min_value=2014, max_value=current_year + 1, value=current_year, step=1)
        history_races = st.slider(
            "Number of previous races to use for prediction",
            min_value=1,
            max_value=8,
            value=3,
        )

        st.caption(
            "The app automatically updates schedules and results whenever new races are added to FastF1."
        )

    # Load and display schedule
    with st.spinner("Loading season schedule from FastF1..."):
        schedule_df = get_event_schedule(year)

    if schedule_df.empty:
        st.error("No schedule data found for this year.")
        return

    # Mark completed vs upcoming races
    schedule_df = schedule_df.copy()
    schedule_df["Status"] = np.where(schedule_df["EventDate"] < today, "Completed", "Upcoming")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"{year} Race Calendar")
        st.dataframe(
            schedule_df.rename(
                columns={
                    "RoundNumber": "Round",
                    "EventName": "Grand Prix",
                    "EventDate": "Date",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Race selection (all races, past and future)
    event_labels = [
        f"Round {int(row.RoundNumber)} - {row.EventName} ({row.EventDate})"
        for _, row in schedule_df.iterrows()
    ]

    with col_right:
        st.subheader("Select Race")
        selected_label = st.selectbox("Choose a race (completed or upcoming)", event_labels)
        selected_round = int(selected_label.split()[1])
        selected_event = schedule_df.loc[schedule_df["RoundNumber"] == selected_round].iloc[0]

        st.markdown(
            f"**Selected:** {selected_event['EventName']}  \n"
            f"**Date:** {selected_event['EventDate']}  \n"
            f"**Status:** {selected_event['Status']}"
        )

        run_prediction = st.button("Run Prediction")

    st.markdown("---")

    if not run_prediction:
        st.info("Select a race and click **Run Prediction** to generate the predicted order.")
        return

    with st.spinner("Building prediction from recent races..."):
        pred_df = build_prediction_for_event(year, selected_round, history_races)

    if pred_df.empty:
        st.warning(
            "Not enough historical race data available to build a prediction for this event. "
            "Try selecting a later round or reducing the number of history races."
        )
        return

    st.subheader("Predicted Finishing Order")
    pred_table = (
        pred_df[["PredictedRank", "Driver", "Team"]]
        .rename(columns={"PredictedRank": "P"})
    )
    st.dataframe(
        pred_table,
        use_container_width=True,
        hide_index=True,
    )

    # If the race is completed, show actual results and a simple comparison
    if selected_event["EventDate"] < today:
        try:
            with st.spinner("Loading actual results for comparison..."):
                actual_df = load_race_results(year, selected_round)
        except Exception as exc:
            st.error(f"Could not load actual results for this race: {exc}")
            return

        actual_df = actual_df.rename(
            columns={
                "Abbreviation": "Driver",
                "TeamName": "Team",
                "Position": "ActualPosition",
            }
        )

        merged = pd.merge(
            pred_df,
            actual_df[["Driver", "ActualPosition"]],
            on="Driver",
            how="left",
        )
        merged["Error"] = merged["ActualPosition"] - merged["PredictedRank"]

        st.subheader("Prediction vs Actual")

        # Side‑by‑side: visual chart and detailed table
        vis_col, table_col = st.columns([1.2, 1])

        valid_for_chart = merged.dropna(subset=["ActualPosition"]).copy()
        if not valid_for_chart.empty:
            chart = (
                alt.Chart(valid_for_chart)
                .mark_circle(size=80, opacity=0.9)
                .encode(
                    x=alt.X("PredictedRank:Q", title="Predicted Position (1 = best)", scale=alt.Scale(reverse=False)),
                    y=alt.Y("ActualPosition:Q", title="Actual Position (1 = best)", scale=alt.Scale(reverse=False)),
                    color=alt.Color("Team:N", legend=None),
                    tooltip=["Driver", "Team", "PredictedRank", "ActualPosition", "Error"],
                )
            )

            # Diagonal reference line (perfect prediction)
            max_pos = int(max(valid_for_chart["PredictedRank"].max(), valid_for_chart["ActualPosition"].max()))
            diag_df = pd.DataFrame({"PredictedRank": list(range(1, max_pos + 1)), "ActualPosition": list(range(1, max_pos + 1))})
            diag = (
                alt.Chart(diag_df)
                .mark_line(strokeDash=[4, 4], color="gray")
                .encode(x="PredictedRank:Q", y="ActualPosition:Q")
            )

            with vis_col:
                st.caption("Dots close to the diagonal line = better predictions.")
                st.altair_chart((chart + diag).properties(height=350), use_container_width=True)

        with table_col:
            merged_table = merged[
                [
                    "PredictedRank",
                    "Driver",
                    "Team",
                    "ActualPosition",
                    "Error",
                ]
            ].rename(
                columns={
                    "PredictedRank": "Pred",
                    "ActualPosition": "Actual",
                }
            )
            st.dataframe(
                merged_table,
                use_container_width=True,
                hide_index=True,
            )

        # Optional compact backtest metrics in an expander
        valid = merged.dropna(subset=["ActualPosition"])
        if not valid.empty:
            with st.expander("Show backtest metrics for this race"):
                mae = (valid["Error"].abs()).mean()

                pred_top3 = set(valid.loc[valid["PredictedRank"] <= 3, "Driver"])
                actual_top3 = set(valid.loc[valid["ActualPosition"] <= 3, "Driver"])
                top3_overlap = len(pred_top3 & actual_top3)

                winner_row = valid.loc[valid["ActualPosition"] == 1]
                winner_in_top3 = False
                if not winner_row.empty:
                    winner_pred_rank = int(winner_row["PredictedRank"].min())
                    winner_in_top3 = winner_pred_rank <= 3

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Mean Abs Error", f"{mae:.2f}")
                col_m2.metric("Top-3 Overlap", f"{top3_overlap}")
                col_m3.metric(
                    "Winner in Pred Top 3",
                    "Yes" if winner_in_top3 else "No",
                )


if __name__ == "__main__":
    main()

