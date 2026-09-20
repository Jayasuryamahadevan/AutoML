from __future__ import annotations

import streamlit as st

from benchmark import predictions_frame, run_ao, run_baseline
from data_pipeline import load_dry_bean

st.set_page_config(page_title="AO Dry Bean Benchmark", layout="wide")
st.title("AO Labs — UCI Dry Bean Benchmark")
st.caption("Seven-class tabular benchmark with deterministic 8-bit feature encoding.")


@st.cache_data(show_spinner="Loading UCI Dry Bean dataset...")
def get_split():
    return load_dry_bean()


split = get_split()

st.write(
    {
        "train_rows": len(split.x_train),
        "test_rows": len(split.x_test),
        "encoded_input_bits": split.x_train.shape[1],
        "classes": list(split.encoder.classes),
    }
)

train_limit = st.slider(
    "Training rows",
    min_value=min(100, len(split.x_train)),
    max_value=len(split.x_train),
    value=min(1000, len(split.x_train)),
    step=100,
)
test_limit = st.slider(
    "Test rows",
    min_value=min(50, len(split.x_test)),
    max_value=len(split.x_test),
    value=min(500, len(split.x_test)),
    step=50,
)
inference_steps = st.number_input("AO inference steps", min_value=1, max_value=20, value=1)

baseline_col, ao_col = st.columns(2)

with baseline_col:
    if st.button("Run Random Forest baseline", use_container_width=True):
        with st.spinner("Running baseline..."):
            result = run_baseline(
                split,
                train_limit=train_limit,
                test_limit=test_limit,
            )
        st.metric("Baseline accuracy", f"{result.accuracy:.2%}")
        frame = predictions_frame(split, result, test_limit=test_limit)
        st.dataframe(frame, use_container_width=True)
        st.download_button(
            "Download baseline predictions",
            frame.to_csv(index=False),
            file_name="dry_bean_random_forest_predictions.csv",
            mime="text/csv",
        )

with ao_col:
    if st.button("Run AO agent", type="primary", use_container_width=True):
        try:
            with st.spinner("Training and evaluating AO agent..."):
                result = run_ao(
                    split,
                    train_limit=train_limit,
                    test_limit=test_limit,
                    inference_steps=int(inference_steps),
                )
        except RuntimeError as exc:
            st.error(str(exc))
            st.info("The AO Labs private-beta ao_core dependency is required for this button.")
        else:
            st.metric("AO accuracy", f"{result.accuracy:.2%}")
            frame = predictions_frame(split, result, test_limit=test_limit)
            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Download AO predictions",
                frame.to_csv(index=False),
                file_name="dry_bean_ao_predictions.csv",
                mime="text/csv",
            )
