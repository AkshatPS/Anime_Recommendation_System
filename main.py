import streamlit as st
import subprocess

# Title of the Streamlit app
st.title("Anime Recommendation System")

# Dropdown to select the recommendation type
option = st.selectbox(
    "Choose Recommendation Model",
    ("Select", "Collaborative Filtering", "Content-Based Filtering")
)

# Based on user selection, run the appropriate script
if option == "Collaborative Filtering":
    st.write("Running Collaborative Filtering...")
    # Run the anime2.py script for Collaborative Filtering
    subprocess.run(["streamlit", "run", "C:/Users/singh/Downloads/final anime/anime2.py"])

elif option == "Content-Based Filtering":
    st.write("Running Content-Based Filtering...")
    # Run the app2.py script for Content-Based Filtering
    subprocess.run(["streamlit", "run", "C:/Users/singh/Downloads/final anime/app2.py"])

else:
    st.write("Please select a filtering method.")
