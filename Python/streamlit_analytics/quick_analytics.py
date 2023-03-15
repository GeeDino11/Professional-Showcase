import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

date = datetime.now().strftime("%m-%d-%Y")



st.title('Quick Analytics')
st.markdown('###### Note: This application is designed for csv and excel files only.')
st.markdown('---')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    try:
        dataframe = pd.read_csv(uploaded_file)

        st.write(dataframe)
        profile = ProfileReport(dataframe, title="Profiling Report")
        export = profile.to_html()

        st.download_button(label="Download Full Report", data=export, file_name=f'Quick Analytics {date}.html', key="1")
        st_profile_report(profile)
        st.download_button(label="Download Full Report", data=export, file_name=f'Quick Analytics {date}.html', key="2", )
        

    except:
        dataframe = pd.read_excel(uploaded_file)

        st.write(dataframe)
        profile = ProfileReport(dataframe, title="Profiling Report")
        export = profile.to_html()

        st.download_button(label="Download Full Report", data=export, file_name=f'Quick Analytics {date}.html', key="1")
        st_profile_report(profile)
        st.download_button(label="Download Full Report", data=export, file_name=f'Quick Analytics {date}', key="2", )
        