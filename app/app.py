import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Abaikan peringatan konvergensi model ARIMA saat runtime Streamlit
warnings.filterwarnings('ignore')

# Set Plotly template for professional look
pio.templates.default = "plotly_white"

# Dictionary for Indonesian Texts
ID_TEXTS = {
    # General & Layout
    "PAGE_TITLE": "Portofolio Analisis Energi",
    "APP_TITLE": "⚡ Analisis Transisi Energi Global: G7 vs. BRICS",
    "APP_SUBTITLE": "Mengukur Laju Dekarbonisasi dan Kesenjangan Pertumbuhan (2000-2022)",
    "ERROR_FILE_NOT_FOUND": "❌ Error: File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.",
    "ERROR_COLUMNS": "❌ Error: Kolom kunci berikut tidak ditemukan di data: {missing_cols}. Harap periksa file Anda.",
    "WARNING_CLEAN_DATA": "⚠️ Peringatan: Data kosong setelah pembersihan. Data untuk G7/BRICS dan rentang tahun 2000-2022 mungkin tidak lengkap.",
    "SUCCESS_MESSAGE": "Analisis ini menunjukkan kemampuan **Data Wrangling, Time Series Analysis, dan Data Storytelling** yang fokus pada implikasi strategis global. 🎉",

    # Sidebar
    "SIDEBAR_TITLE": "👤 Portofolio Analisis",
    "SIDEBAR_SUBHEADER": "M Feby Khoiru Sidqi\n/ AI Research Enthusiast / Data Engineer / Data Scientist/ Python Developer",
    "SIDEBAR_ANALYSIS_CAPTION": "Analisis: Transisi Energi Global G7 vs BRICS",
    "SIDEBAR_TOOLS_CAPTION": "Metodologi & Alat:",
    "SIDEBAR_TOOLS_FOCUS": "Fokus: Time Series Analysis, Data Storytelling",
    "SIDEBAR_TOOLS_DATA": "Data: OWID Energy Data (2000-2022)",
    "SIDEBAR_CONTACT_CAPTION": "Kontak & Profil",

    # Executive Summary
    "EXEC_TITLE": "Ringkasan Eksekutif: Pergeseran Kekuatan Energi",
    "EXEC_SUBHEADER_1": "Pangsa Fosil",
    "EXEC_SUBHEADER_2": "Konsumsi Absolut",
    "EXEC_SUBHEADER_3": "Dominasi EBT",
    "EXEC_DELTA_1": "Menurun Berkelanjutan",
    "EXEC_DELTA_2": "Konsumsi Absolut Jauh Lebih Tinggi",
    "EXEC_DELTA_3": "Akselerasi Kuat",
    "EXEC_CAPTION_1": "G7 memiliki pangsa fosil rata-rata yang lebih rendah.",
    "EXEC_CAPTION_2": "BRICS telah melampaui G7 dalam total konsumsi fosil sekitar tahun **{}**.",
    "EXEC_CAPTION_3": "BRICS mendominasi pertumbuhan kapasitas EBT baru.",

    # Deep Dive
    "DEEP_DIVE_TITLE": "Analisis Mendalam: Bukti Transisi dan Tantangan",

    # Plot 1: Fossil Share Trend
    "PLOT_1_HEADER": "1. Tren Rata-Rata Pangsa Energi Fosil (%)",
    "PLOT_1_INSIGHT": "Insight: G7 mempertahankan pangsa yang relatif stabil (tetap tinggi), sementara **BRICS menunjukkan laju penurunan yang lebih agresif** pasca 2010.",

    # Plot 2: Fossil Absolute Trend
    "PLOT_2_HEADER": "2. Total Konsumsi Energi Fosil (TWh): BRICS Melampaui G7",
    "PLOT_2_INSIGHT": "Insight: BRICS telah meningkatkan **volume absolut** konsumsi fosil secara masif, **melampaui G7 sekitar tahun {}** untuk mendukung industrialisasi.",

    # Plot 3: Renewable Growth
    "PLOT_3_HEADER": "3. Pertumbuhan Absolut Energi Surya & Angin (2012-2022)",
    "PLOT_3_INSIGHT": "Insight: **China (BRICS)** adalah mesin pertumbuhan energi terbarukan absolut, jauh melampaui negara-negara G7.",

    # Plot 4: Energy Efficiency
    "PLOT_4_HEADER": "4. Tren Intensitas Energi (Energy per GDP)",
    "PLOT_4_INSIGHT": "Insight: BRICS menunjukkan **penurunan yang lebih curam** dibandingkan G7, menandai peningkatan efisiensi makroekonomi yang signifikan.",

    # Plot 5: Low Carbon Share
    "PLOT_5_HEADER": "5. Pangsa Energi Rendah Karbon (Low Carbon Share)",
    "PLOT_5_INSIGHT_1": "Narasi Kunci (Dominasi Bersih):",
    "PLOT_5_INSIGHT_2": "* **Kepemimpinan G7:** G7 secara historis memiliki pangsa energi rendah karbon yang lebih tinggi (nuklir, hidro).",
    "PLOT_5_INSIGHT_3": "* **Kesenjangan yang Cepat Terkejar:** BRICS menunjukkan laju pertumbuhan *low carbon share* yang cepat, didorong oleh investasi EBT baru.",

    # Plot 6: ARIMA Forecast
    "PLOT_6_HEADER": "6. Proyeksi Pangsa Fosil Rata-rata (%) dengan ARIMA hingga 2030",
    "PLOT_6_WARNING_ARIMA": "⚠️ Warning: Gagal melatih model ARIMA. Error: {e}",
    "PLOT_6_METRIC_G7": "Akurasi Prediksi G7 (MAPE)",
    "PLOT_6_METRIC_BRICS": "Akurasi Prediksi BRICS (MAPE)",
    "PLOT_6_METRIC_HELP": "Mean Absolute Percentage Error (Semakin kecil semakin baik)",
    "PLOT_6_NARRATIVE_1": "Narasi Kunci (Proyeksi Time Series):",
    "PLOT_6_NARRATIVE_2": "Proyeksi Jangka Panjang: Model memprediksi **penurunan Pangsa Fosil G7 berlanjut hingga 2030**, sementara BRICS menunjukkan **penurunan yang lebih landai** atau stabil, menyoroti tantangan besar dalam mencapai target dekarbonisasi global.",
    
    # NEW SECTION: Theoretical Grounding
    "THEORY_TITLE": "7. Pengembangan Teori: Sintesis dari Perspektif Internasional",
    "THEORY_SUBTITLE": "Hasil analisis ini sejalan dengan tiga narasi utama dalam literatur transisi energi global:",
    
    "THEORY_POINT_1_TITLE": "1. Dekopling (Decoupling) dan Efisiensi",
    "THEORY_POINT_1_BODY": "Penurunan intensitas energi (Plot 4) di kedua blok sejalan dengan konsep **Decoupling**. Meskipun G7 menunjukkan dekopling **absolut** (penurunan emisi/konsumsi absolut), BRICS menunjukkan laju dekopling **relatif** (peningkatan efisiensi) yang lebih cepat, sebagaimana dibahas dalam studi oleh **IEA dan Nature Energy**, yang menyoroti bahwa BRICS masih berjuang dengan kenaikan konsumsi absolut demi pembangunan ekonomi.",
    
    "THEORY_POINT_2_TITLE": "2. Paradigma 'Tantangan Ganda' (Dual Challenge)",
    "THEORY_POINT_2_BODY": "Crossover BRICS di Konsumsi Fosil Absolut (Plot 2) mengkonfirmasi adanya **Tantangan Ganda**: BRICS harus memenuhi kebutuhan energi untuk pembangunan dan pertumbuhan populasi yang cepat sambil mendekarbonisasi. Literatur **Just Transition** menekankan bahwa negara berkembang tidak dapat mengorbankan pembangunan, yang menjelaskan peningkatan volume absolut konsumsi fosil mereka meskipun pangsa EBT mereka meningkat.",

    "THEORY_POINT_3_TITLE": "3. Geopolitik Energi Terbarukan (Renewable Energy Geopolitics)",
    "THEORY_POINT_3_BODY": "Dominasi China (BRICS) dalam pertumbuhan Solar/Wind (Plot 3) mencerminkan pergeseran geopolitik. Jurnal-jurnal **Energy Policy** menunjukkan bahwa konsentrasi manufaktur EBT di BRICS/China mempercepat transisi global namun menciptakan **ketergantungan energi baru**—bergeser dari minyak/gas ke teknologi EBT. China kini menjadi 'OPEC' dalam rantai pasok energi bersih.",
}

# Dictionary for English Texts
EN_TEXTS = {
    # General & Layout
    "PAGE_TITLE": "Energy Analysis Portfolio",
    "APP_TITLE": "⚡ Global Energy Transition Analysis: G7 vs. BRICS",
    "APP_SUBTITLE": "Measuring Decarbonization Pace and Growth Gap (2000-2022)",
    "ERROR_FILE_NOT_FOUND": "❌ Error: File '{file_path}' not found. Ensure the file is in the same directory as app.py.",
    "ERROR_COLUMNS": "❌ Error: The following key columns were not found in the data: {missing_cols}. Please check your file.",
    "WARNING_CLEAN_DATA": "⚠️ Warning: Data is empty after cleaning. Data for G7/BRICS and the 2000-2022 range may be incomplete.",
    "SUCCESS_MESSAGE": "This analysis demonstrates skills in **Data Wrangling, Time Series Analysis, and Data Storytelling** focused on global strategic implications. 🎉",

    # Sidebar
    "SIDEBAR_TITLE": "👤 Analysis Portfolio",
    "SIDEBAR_SUBHEADER": "M Feby Khoiru Sidqi\n AI Research Enthusiast/ Data Engineer / Data Scientist/ Python Developer",
    "SIDEBAR_ANALYSIS_CAPTION": "Analysis: G7 vs BRICS Global Energy Transition",
    "SIDEBAR_TOOLS_CAPTION": "Methodology & Tools:",
    "SIDEBAR_TOOLS_FOCUS": "Focus: Time Series Analysis, Data Storytelling",
    "SIDEBAR_TOOLS_DATA": "Data: OWID Energy Data (2000-2022)",
    "SIDEBAR_CONTACT_CAPTION": "Contact & Profiles",

    # Executive Summary
    "EXEC_TITLE": "Executive Summary: Shifting Energy Power",
    "EXEC_SUBHEADER_1": "Fossil Share",
    "EXEC_SUBHEADER_2": "Absolute Consumption",
    "EXEC_SUBHEADER_3": "Renewables Dominance",
    "EXEC_DELTA_1": "Sustainably Declining",
    "EXEC_DELTA_2": "Much Higher Absolute Consumption",
    "EXEC_DELTA_3": "Strong Acceleration",
    "EXEC_CAPTION_1": "G7 has a lower average fossil fuel share.",
    "EXEC_CAPTION_2": "BRICS surpassed G7 in total fossil fuel consumption around the year **{}**.",
    "EXEC_CAPTION_3": "BRICS dominates new renewable energy capacity growth.",

    # Deep Dive
    "DEEP_DIVE_TITLE": "Deep Dive Analysis: Transition Evidence and Challenges",

    # Plot 1: Fossil Share Trend
    "PLOT_1_HEADER": "1. Average Fossil Energy Share Trend (%)",
    "PLOT_1_INSIGHT": "Insight: G7 maintains a relatively stable (still high) share, while **BRICS shows a more aggressive rate of decline** post-2010.",

    # Plot 2: Fossil Absolute Trend
    "PLOT_2_HEADER": "2. Total Fossil Energy Consumption (TWh): BRICS Surpasses G7",
    "PLOT_2_INSIGHT": "Insight: BRICS has massively increased the **absolute volume** of fossil consumption, **surpassing G7 around {}** to support industrialization.",

    # Plot 3: Renewable Growth
    "PLOT_3_HEADER": "3. Absolute Growth of Solar & Wind Energy (2012-2022)",
    "PLOT_3_INSIGHT": "Insight: **China (BRICS)** is the engine of absolute renewable energy growth, significantly outpacing G7 nations.",

    # Plot 4: Energy Efficiency
    "PLOT_4_HEADER": "4. Energy Intensity Trend (Energy per GDP)",
    "PLOT_4_INSIGHT": "Insight: BRICS shows a **steeper decline** compared to G7, signaling significant macroeconomic efficiency improvements.",

    # Plot 5: Low Carbon Share
    "PLOT_5_HEADER": "5. Low Carbon Energy Share (The Sustainability Test)",
    "PLOT_5_INSIGHT_1": "Key Narrative (Clean Dominance):",
    "PLOT_5_INSIGHT_2": "* **G7 Leadership:** G7 historically possesses a higher low-carbon energy share (nuclear, hydro).",
    "PLOT_5_INSIGHT_3": "* **Rapidly Closing Gap:** BRICS shows a fast growth rate in *low carbon share*, driven by new RE investments.",

    # Plot 6: ARIMA Forecast
    "PLOT_6_HEADER": "6. Projected Average Fossil Share (%) with ARIMA until 2030",
    "PLOT_6_WARNING_ARIMA": "⚠️ Warning: Failed to train ARIMA model. Error: {e}",
    "PLOT_6_METRIC_G7": "G7 Prediction Accuracy (MAPE)",
    "PLOT_6_METRIC_BRICS": "BRICS Prediction Accuracy (MAPE)",
    "PLOT_6_METRIC_HELP": "Mean Absolute Percentage Error (Lower is better)",
    "PLOT_6_NARRATIVE_1": "Key Narrative (Time Series Projection):",
    "PLOT_6_NARRATIVE_2": "Long-Term Projection: The model predicts **G7's Fossil Share decline continues until 2030**, while BRICS shows a **flatter** or stable decline, highlighting the significant challenge in meeting global decarbonization targets.",
    
    # NEW SECTION: Theoretical Grounding
    "THEORY_TITLE": "7. Theory Development: A Synthesis from an International Perspective",
    "THEORY_SUBTITLE": "This analysis aligns with three major narratives in global energy transition literature:",

    "THEORY_POINT_1_TITLE": "1. Decoupling and Efficiency",
    "THEORY_POINT_1_BODY": "The decline in energy intensity (Plot 4) across both blocs is consistent with the **Decoupling** concept. While G7 is associated with **absolute** decoupling (falling absolute emissions/consumption), BRICS shows a faster rate of **relative** decoupling (efficiency gains), as discussed in studies by the **IEA and Nature Energy**, which note that BRICS still struggles with rising absolute consumption for economic development.",

    "THEORY_POINT_2_TITLE": "2. The 'Dual Challenge' Paradigm",
    "THEORY_POINT_2_BODY": "The BRICS crossover in Absolute Fossil Consumption (Plot 2) confirms the **Dual Challenge**: BRICS must meet the energy needs for rapid development and population growth while decarbonizing. **Just Transition** literature emphasizes that developing nations cannot sacrifice development, which explains the rising absolute volume of their fossil consumption despite increasing renewable shares.",

    "THEORY_POINT_3_TITLE": "3. Renewable Energy Geopolitics",
    "THEORY_POINT_3_BODY": "China's (BRICS) dominance in Solar/Wind growth (Plot 3) reflects the shift in geopolitics. **Energy Policy** journals suggest that the concentration of RE manufacturing in BRICS/China accelerates the global transition but creates **new energy dependencies**—shifting from oil/gas to RE technology supply chains. China is becoming the 'OPEC' of clean energy supply.",
}


def get_texts(lang_choice):
    """Mengembalikan dictionary teks yang sesuai berdasarkan pilihan bahasa."""
    if lang_choice == 'Bahasa Indonesia':
        return ID_TEXTS
    else:
        return EN_TEXTS

# --- 1. DATA LOADING AND CLEANING FUNCTIONS ---

@st.cache_data
def load_and_clean_data(file_path, texts):
    """Memuat, membersihkan, dan menyiapkan data untuk analisis G7 vs BRICS."""
    
    # 1. Definisi Parameter (TETAP)
    g7_countries = ['United States', 'Canada', 'Germany', 'United Kingdom', 'France', 'Italy', 'Japan']
    brics_countries = ['Brazil', 'Russia', 'India', 'China', 'South Africa']
    target_countries = g7_countries + brics_countries
    start_year = 2000
    end_year = 2022
    
    key_columns = [
        'country', 'year', 'gdp', 'population', 
        'fossil_share_energy', 'low_carbon_share_energy', 
        'solar_consumption', 'wind_consumption', 
        'energy_per_gdp', 'fossil_fuel_consumption',
        'renewables_consumption'
    ]

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(texts["ERROR_FILE_NOT_FOUND"].format(file_path=file_path))
        return None, end_year 

    # 2. Filter Negara Target dan Tahun
    df_filtered = df[
        (df['country'].isin(target_countries)) & 
        (df['year'] >= start_year) & 
        (df['year'] <= end_year)
    ].copy()

    # Cek ketersediaan kolom
    missing_cols = [col for col in key_columns if col not in df_filtered.columns]
    if missing_cols:
        st.error(texts["ERROR_COLUMNS"].format(missing_cols=', '.join(missing_cols)))
        return None, end_year

    # 3. Seleksi Kolom Kunci & Interpolasi
    df_clean = df_filtered[key_columns].copy()
    df_clean['gdp'] = df_clean.groupby('country')['gdp'].transform(lambda x: x.interpolate(method='linear'))

    # 4. Membuat Kolom Kelompok Negara
    def assign_group(country):
        return 'G7' if country in g7_countries else 'BRICS'
    df_clean['Group'] = df_clean['country'].apply(assign_group)

    # 5. Membuang sisa NaN pada kolom utama
    subset_cols = ['gdp', 'fossil_share_energy', 'low_carbon_share_energy', 'energy_per_gdp', 'fossil_fuel_consumption']
    df_clean.dropna(subset=subset_cols, inplace=True)
    
    if df_clean.empty:
        st.error(texts["WARNING_CLEAN_DATA"])
        return None, end_year
        
    return df_clean, end_year

# --- 2. VISUALIZATION FUNCTIONS ---

def plot_fossil_share_trend(df, texts):
    """Plot 1: Tren Pangsa Fosil (Persentase)"""
    st.header(texts["PLOT_1_HEADER"])
    df_group_trend = df.groupby(['Group', 'year'])['fossil_share_energy'].mean().reset_index()
    # Plotly titles and labels remain English for clarity in the visual component
    fig = px.line(df_group_trend, x='year', y='fossil_share_energy', color='Group', labels={'fossil_share_energy': 'Fossil Share (%)', 'year': 'Year'}, markers=True, color_discrete_map={'G7': '#347C98', 'BRICS': '#E36414'})
    fig.update_layout(title_text="Average Fossil Energy Share Trend (2000-2022)", title_x=0.5, yaxis_title="Fossil Share (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(texts["PLOT_1_INSIGHT"])
    st.markdown("---")

def plot_fossil_absolute_trend(df, latest_year, crossover_year, texts):
    """Plot 2: Konsumsi Fosil Absolut (TWh) - Executive Grade"""
    st.header(texts["PLOT_2_HEADER"])
    df_fossil_absolute_trend = df.groupby(['Group', 'year'])['fossil_fuel_consumption'].sum().reset_index()
    # Plotly titles and labels remain English
    fig = px.line(df_fossil_absolute_trend, x='year', y='fossil_fuel_consumption', color='Group', labels={'fossil_fuel_consumption': 'Total Fossil Consumption (TWh)', 'year': 'Year'}, markers=True, line_shape='spline', color_discrete_map={'G7': '#347C98', 'BRICS': '#E36414'})
    if crossover_year and crossover_year > df['year'].min():
        fig.add_vline(x=crossover_year, line_dash="dash", line_color="#7A7A7A", annotation_text=f"Crossover ({int(crossover_year)})", annotation_position="top left", annotation_font_color="#7A7A7A")
    fig.update_layout(title_text="Total Fossil Energy Consumption (TWh)", title_x=0.5, yaxis_tickformat=',.2s', hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(texts["PLOT_2_INSIGHT"].format(int(crossover_year)))
    st.markdown("---")


def plot_renewable_growth(df, texts):
    """Plot 3: Pertumbuhan Energi Terbarukan Absolut (Solar & Wind)"""
    st.header(texts["PLOT_3_HEADER"])
    df_growth = df[(df['year'] == 2012) | (df['year'] == df['year'].max())].copy()
    df_pivot = df_growth.pivot_table(index=['country', 'Group'], columns='year', values=['solar_consumption', 'wind_consumption']).reset_index()
    
    # Perbaikan Syntax (Handling multi-level columns after pivot)
    df_pivot.columns = ['_'.join(map(str, col)).strip('_') if col[1] != '' else col[0] for col in df_pivot.columns.values]
    
    latest_year = df['year'].max()
    df_pivot['Solar_Growth_TWh'] = df_pivot[f'solar_consumption_{latest_year}'] - df_pivot['solar_consumption_2012']
    df_pivot['Wind_Growth_TWh'] = df_pivot[f'wind_consumption_{latest_year}'] - df_pivot['wind_consumption_2012']
    
    df_growth_melt = df_pivot.melt(id_vars=['country', 'Group'], value_vars=['Solar_Growth_TWh', 'Wind_Growth_TWh'], var_name='Source', value_name='Growth_TWh')
    df_growth_melt['Source'] = df_growth_melt['Source'].str.replace('_Growth_TWh', '')

    # Plotly titles and labels remain English
    fig = px.bar(
        df_growth_melt.sort_values(['Growth_TWh'], ascending=False),
        x='country',
        y='Growth_TWh',
        color='Source',
        facet_col='Source',
        facet_col_wrap=2,
        labels={'Growth_TWh': 'Consumption Growth (TWh)', 'country': 'Country'},
        text_auto='.3s',
        color_discrete_map={'Solar': '#FFC300', 'Wind': '#4CAF50'}
    )
    fig.update_layout(title_text="Absolute Growth of Solar and Wind Energy (2012-2022)", title_x=0.5, yaxis_title="Growth (TWh)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(texts["PLOT_3_INSIGHT"])
    st.markdown("---")


def plot_energy_efficiency(df, texts):
    """Plot 4: Efisiensi Energi (Energy per GDP)"""
    st.header(texts["PLOT_4_HEADER"])
    df_efficiency_trend = df.groupby(['Group', 'year'])['energy_per_gdp'].mean().reset_index()
    # Plotly titles and labels remain English
    fig = px.line(df_efficiency_trend, x='year', y='energy_per_gdp', color='Group', labels={'energy_per_gdp': 'Energy per GDP (koe/$) - Lower is More Efficient', 'year': 'Year'}, markers=True, color_discrete_map={'G7': '#347C98', 'BRICS': '#E36414'})
    fig.update_layout(title_text="Energy Intensity Trend (Energy per GDP)", title_x=0.5, yaxis_title="Energy Intensity (koe/$)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(texts["PLOT_4_INSIGHT"])
    st.markdown("---")

def plot_low_carbon_share(df, texts):
    """Plot 5: Pangsa Energi Rendah Karbon (The Sustainability Test)"""
    st.header(texts["PLOT_5_HEADER"])
    df_low_carbon = df.groupby(['Group', 'year'])['low_carbon_share_energy'].mean().reset_index()
    
    # Plotly titles and labels remain English
    fig = px.line(
        df_low_carbon,
        x='year',
        y='low_carbon_share_energy',
        color='Group',
        title='Low Carbon Energy Share (%)',
        labels={'low_carbon_share_energy': 'Low Carbon Energy Share (%)', 'year': 'Year'},
        markers=True,
        color_discrete_map={'G7': '#347C98', 'BRICS': '#E36414'}
    )
    fig.update_layout(title_x=0.5, yaxis_title="Low Carbon Share (%)", yaxis_range=[0, df_low_carbon['low_carbon_share_energy'].max() * 1.1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(texts["PLOT_5_INSIGHT_1"])
    st.markdown(texts["PLOT_5_INSIGHT_2"])
    st.markdown(texts["PLOT_5_INSIGHT_3"])
    st.markdown("---")


def plot_fossil_share_forecast_arima(df, texts):
    """Plot 6: Prediksi Pangsa Fosil Rata-rata (Menggunakan Model ARIMA) hingga 2030."""
    
    st.header(texts["PLOT_6_HEADER"])
    
    df_fossil_share = df.groupby(['Group', 'year'])['fossil_share_energy'].mean().reset_index()
    
    test_size = 3
    
    ts_g7_train = df_fossil_share[df_fossil_share['Group'] == 'G7'].set_index('year')['fossil_share_energy'][:-test_size]
    ts_g7_test = df_fossil_share[df_fossil_share['Group'] == 'G7'].set_index('year')['fossil_share_energy'][-test_size:]
    
    ts_brics_train = df_fossil_share[df_fossil_share['Group'] == 'BRICS'].set_index('year')['fossil_share_energy'][:-test_size]
    ts_brics_test = df_fossil_share[df_fossil_share['Group'] == 'BRICS'].set_index('year')['fossil_share_energy'][-test_size:]
    
    forecast_periods = 8 

    mape_g7, mape_brics = 999, 999
    
    try:
        # G7
        model_g7 = ARIMA(ts_g7_train, order=(1, 1, 0)).fit()
        g7_pred_test = model_g7.predict(start=ts_g7_test.index.min(), end=ts_g7_test.index.max())
        forecast_g7 = model_g7.predict(start=ts_g7_train.index.max() + 1, end=ts_g7_train.index.max() + forecast_periods)
        mape_g7 = mean_absolute_percentage_error(ts_g7_test, g7_pred_test) * 100

        # BRICS
        model_brics = ARIMA(ts_brics_train, order=(1, 1, 0)).fit()
        brics_pred_test = model_brics.predict(start=ts_brics_test.index.min(), end=ts_brics_test.index.max())
        forecast_brics = model_brics.predict(start=ts_brics_train.index.max() + 1, end=ts_brics_train.index.max() + forecast_periods)
        mape_brics = mean_absolute_percentage_error(ts_brics_test, brics_pred_test) * 100

    except Exception as e:
        st.warning(texts["PLOT_6_WARNING_ARIMA"].format(e=e))
        forecast_g7 = pd.Series([ts_g7_train.iloc[-1]] * forecast_periods, index=range(ts_g7_train.index.max() + 1, ts_g7_train.index.max() + forecast_periods + 1))
        forecast_brics = pd.Series([ts_brics_train.iloc[-1]] * forecast_periods, index=range(ts_brics_train.index.max() + 1, ts_brics_train.index.max() + forecast_periods + 1))


    # Gabungkan data untuk visualisasi
    df_g7_plot = pd.concat([ts_g7_train, ts_g7_test, forecast_g7]).rename('G7').to_frame().reset_index().rename(columns={'index': 'year'})
    df_brics_plot = pd.concat([ts_brics_train, ts_brics_test, forecast_brics]).rename('BRICS').to_frame().reset_index().rename(columns={'index': 'year'})
    
    df_combined_plot = pd.merge(df_g7_plot, df_brics_plot, on='year', how='outer').melt(
        id_vars='year', var_name='Group', value_name='Fossil_Share_Pct'
    )
    
    df_combined_plot['Type'] = np.select(
        [
            df_combined_plot['year'] <= ts_g7_test.index.max(),
            df_combined_plot['year'] > ts_g7_test.index.max()
        ],
        [
            'Historical & Validation', # Tetap English di kolom dataframe
            'Projection (ARIMA)'      # Tetap English di kolom dataframe
        ],
        default='Historical & Validation'
    )
    
    # --- PLOTLY INTERAKTIF (Labels and Titles are English) ---
    fig = go.Figure()

    for group, color in [('G7', '#347C98'), ('BRICS', '#E36414')]:
        df_hist = df_combined_plot[(df_combined_plot['Group'] == group) & (df_combined_plot['Type'] == 'Historical & Validation')]
        fig.add_trace(go.Scatter(x=df_hist['year'], y=df_hist['Fossil_Share_Pct'], 
                                 mode='lines+markers', name=f'{group} (Historical)', 
                                 line=dict(color=color, width=3)))

    for group, color in [('G7', '#347C98'), ('BRICS', '#E36414')]:
        df_pred = df_combined_plot[(df_combined_plot['Group'] == group) & (df_combined_plot['Type'] == 'Projection (ARIMA)')]
        fig.add_trace(go.Scatter(x=df_pred['year'], y=df_pred['Fossil_Share_Pct'], 
                                 mode='lines+markers', name=f'{group} (Projection)', 
                                 line=dict(color=color, dash='dash', width=2), 
                                 marker=dict(symbol='diamond')))

    fig.update_layout(
        title='Projected Average Fossil Share (%) until 2030',
        xaxis_title='Year',
        yaxis_title='Average Fossil Share (%)',
        title_x=0.5, 
        yaxis_range=[df_combined_plot['Fossil_Share_Pct'].min() * 0.9, df_combined_plot['Fossil_Share_Pct'].max() * 1.1],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    col_mape_g7, col_mape_brics = st.columns(2)
    with col_mape_g7:
        st.metric(label=texts["PLOT_6_METRIC_G7"], value=f"{mape_g7:.2f}%", help=texts["PLOT_6_METRIC_HELP"])
    with col_mape_brics:
        st.metric(label=texts["PLOT_6_METRIC_BRICS"], value=f"{mape_brics:.2f}%", help=texts["PLOT_6_METRIC_HELP"])
        
    st.markdown(texts["PLOT_6_NARRATIVE_1"])
    st.markdown(texts["PLOT_6_NARRATIVE_2"])
    st.markdown("---")

# --- 3. THEORETICAL GROUNDING ---
def add_theoretical_grounding(texts):
    """Menambahkan bagian yang membahas konteks teoritis dari jurnal internasional."""
    st.header(texts["THEORY_TITLE"])
    st.subheader(texts["THEORY_SUBTITLE"])
    st.markdown("---")

    # Point 1: Decoupling
    st.markdown(f"**{texts['THEORY_POINT_1_TITLE']}**")
    st.markdown(texts["THEORY_POINT_1_BODY"])
    st.markdown("")
    
    # Point 2: Dual Challenge
    st.markdown(f"**{texts['THEORY_POINT_2_TITLE']}**")
    st.markdown(texts["THEORY_POINT_2_BODY"])
    st.markdown("")

    # Point 3: Renewable Geopolitics
    st.markdown(f"**{texts['THEORY_POINT_3_TITLE']}**")
    st.markdown(texts["THEORY_POINT_3_BODY"])
    st.markdown("---")


# --- 4. STREAMLIT APP LAYOUT ---

def main():
    # Load default language or state
    if 'language' not in st.session_state:
        st.session_state.language = 'Bahasa Indonesia'

    texts = get_texts(st.session_state.language)

    # Streamlit Page Config
    st.set_page_config(layout="wide", page_title=texts["PAGE_TITLE"])
    
    # --- LAYOUT PORTFOLIO DI SIDEBAR ---
    with st.sidebar:
        # Language Selector
        st.subheader("🌐 Language / Bahasa")
        lang_choice = st.radio(
            "Pilih Bahasa / Choose Language",
            ('Bahasa Indonesia', 'English'),
            key='lang_selector',
            index=0 if st.session_state.language == 'Bahasa Indonesia' else 1,
            horizontal=True
        )
        if lang_choice != st.session_state.language:
            st.session_state.language = lang_choice
            st.rerun() # Rerun to apply new language texts

        st.markdown("---")
        st.title(texts["SIDEBAR_TITLE"])
        # Placeholder for profile image (Assuming 'img/profil.jpg' is available in the environment)
        st.image("https://avatars.githubusercontent.com/u/113446269?v=4", caption=texts["SIDEBAR_SUBHEADER"]) 
        st.markdown("---")
        st.caption(texts["SIDEBAR_ANALYSIS_CAPTION"])
        st.caption(texts["SIDEBAR_TOOLS_CAPTION"])
        st.markdown(f"* **Alat:** Python (Pandas, Streamlit, Plotly, **ARIMA**)")
        st.markdown(f"* **{texts['SIDEBAR_TOOLS_FOCUS']}**")
        st.markdown(f"* **{texts['SIDEBAR_TOOLS_DATA']}**")
        st.markdown("---")
        st.caption(texts["SIDEBAR_CONTACT_CAPTION"])
        st.markdown("https://www.linkedin.com/in/m-feby-khoiru-sidqi")
        st.markdown("https://github.com/mfebykhoirusidqi/World-Energy-Consumption")
        
    
    st.title(texts["APP_TITLE"])
    st.markdown(f"### {texts['APP_SUBTITLE']}")
    st.markdown("---")

    # Load Data 
    file_path = "data/owid-energy-data.csv" 
    data_tuple = load_and_clean_data(file_path, texts)
    
    if data_tuple[0] is None:
        return 
        
    df_clean, latest_year = data_tuple

    # --- HITUNG METRIK KUNCI UNTUK RINGKASAN ---
    df_fossil_absolute_trend = df_clean.groupby(['Group', 'year'])['fossil_fuel_consumption'].sum().reset_index()
    df_crossover = df_fossil_absolute_trend.pivot(index='year', columns='Group', values='fossil_fuel_consumption').reset_index()
    crossover_year_series = df_crossover[df_crossover['BRICS'] > df_crossover['G7']]['year']
    crossover_year = int(crossover_year_series.min()) if not crossover_year_series.empty else latest_year
    
    brics_cons_2022 = df_fossil_absolute_trend[
        (df_fossil_absolute_trend['Group'] == 'BRICS') & 
        (df_fossil_absolute_trend['year'] == latest_year)
    ]['fossil_fuel_consumption'].sum()

    # --- RINGKASAN EKSEKUTIF ---
    st.header(texts["EXEC_TITLE"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(texts["EXEC_SUBHEADER_1"])
        g7_share = df_clean[(df_clean['Group'] == 'G7') & (df_clean['year'] == latest_year)]['fossil_share_energy'].mean()
        st.metric(label=f"G7 ({latest_year})", value=f"{g7_share:.1f}%", delta=texts["EXEC_DELTA_1"])
        st.caption(texts["EXEC_CAPTION_1"])
        
    with col2:
        st.subheader(texts["EXEC_SUBHEADER_2"])
        # Format dalam ribu TWh (K TWh)
        st.metric(label=f"BRICS ({latest_year})", value=f"{brics_cons_2022/1000:,.1f}K TWh", delta=texts["EXEC_DELTA_2"])
        st.caption(texts["EXEC_CAPTION_2"].format(crossover_year))
        
    with col3:
        st.subheader(texts["EXEC_SUBHEADER_3"])
        brics_low_carbon = df_clean[(df_clean['Group'] == 'BRICS') & (df_clean['year'] == latest_year)]['low_carbon_share_energy'].mean()
        st.metric(label="Low Carbon Share", value=f"BRICS: {brics_low_carbon:.1f}%", delta=texts["EXEC_DELTA_3"])
        st.caption(texts["EXEC_CAPTION_3"])

    st.markdown("---")
    
    # --- VISUALISASI DAN NARASI MENDALAM ---
    st.header(texts["DEEP_DIVE_TITLE"])
    
    # Semua plot kini menggunakan string dari dictionary 'texts'
    plot_fossil_share_trend(df_clean, texts)
    plot_energy_efficiency(df_clean, texts)
    plot_fossil_absolute_trend(df_clean, latest_year, crossover_year, texts)
    plot_renewable_growth(df_clean, texts)
    plot_low_carbon_share(df_clean, texts) 
    plot_fossil_share_forecast_arima(df_clean, texts)
    
    # --- TEORI BARU DAN KESIMPULAN DI SINI ---
    add_theoretical_grounding(texts)
    
    st.success(texts["SUCCESS_MESSAGE"])

if __name__ == "__main__":
    main()