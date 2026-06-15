
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import seaborn as sns
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Emisi Karbon",
    page_icon= "🚗"
)
st.title("Emisi Karbon Pada Aktivitas Logistik E-commerce")
st.write("Dashboard ini berisi sebaran data dan alat prediksi jumlah emisi karbon yang dihasilkan aktivitas logistik E-Commerce")


# Load Model Random Forest yang sudah dilatih 
model = joblib.load("randomforest.pkl")

# Load dataset
df = pd.read_csv('ecommerce_logistics_carbon_emissions_v1.csv')

# Hapus kolom yang tidak relevan
df.drop(columns=['Transaction_ID', 'Date', 'Destination_City', 'Is_Eco_Friendly'], inplace=True)

# Handle Outlier
for col in df.select_dtypes(include='number').columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
        
    batas_atas = Q3 + 1.5 * IQR
    batas_bawah = Q1 - 1.5 * IQR
        
        
    df = df[(df[col] >= batas_bawah) & (df[col] <= batas_atas)]

# Ambil nilai unik pada kolom Origin_Facility, Vehicle_Type, Route_Type
origin_options = sorted(df['Origin_Facility'].unique())
vehicle_options = sorted(df['Vehicle_Type'].unique())
route_options = sorted(df['Route_Type'].unique())
        
# Mengubah tipe data pada kolom akategori
traffic_map = {
    'Low': 0,
    'Normal': 1,
    'High': 2,
    'Severe Congestion': 3
}
df['Traffic_Conditions'] = df['Traffic_Conditions'].map(traffic_map)

df = pd.get_dummies(df, columns=['Origin_Facility', 'Vehicle_Type', 'Route_Type'])
   
# Pisahkan variabel dependen dan independen
x = df.drop(columns=['Carbon_Emission_kgCO2e'])
y = df['Carbon_Emission_kgCO2e']

# Pisah data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standarisasi Variabel Dependen
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# Buat prediksi menggunakan model yang sudah dilatih
y_pred = model.predict(x_test_scaled)

# itung evaluasi model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
    
tab1, tab2 = st.tabs([
    "Prediksi",
    "Sebaran Data"
])

with tab1:
    st.header('Prediksi Jumlah Emisi Karbon')
    
    st.subheader("Evaluasi Model")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="MAE", value=f"{mae:.2f}")
    with col2:
        st.metric(label="MSE", value=f"{mse:.2f}")
    with col3:
        st.metric(label="RMSE", value=f"{rmse:.2f}")
    with col4:
        st.metric(label="R2", value=f"{r2:.3f}") 

    original_data = pd.read_csv("ecommerce_logistics_carbon_emissions_v1.csv")
    original_data = original_data.dropna(subset=['Transaction_ID', 'Date', 'Destination_City', 'Is_Eco_Friendly'])

    st.subheader("Form Prediksi")
    st.write("Isi semua kolom pada form di bawah ini dan tekan tombol 'Prediksi' intuk melihat hasil predisi jumlah emisi karbon")
    # Buat form input untuk data prediksi
    with st.form('prediction_form'):
        Origin_Facility = st.selectbox('Masukan Tempat Keberangkatan:', options=origin_options)
        Vehicle_Type = st.selectbox('Masukan Jenis Kendaraan:', options=vehicle_options)
        Route_Type = st.selectbox('Masukan Rute Perjalanan:', options=route_options)
        Distance_KM = st.number_input('Masukan Jarak Perjalanan (KM):',min_value=0.0)
        Package_Weight_KG = st.number_input('Masukan Berat Angkutan (Kg):', min_value=0.0)
        Traffic_Conditions = st.selectbox('Masukan Kondisi Lalu Lintas:', options=list(traffic_map.keys()))

        submit = st.form_submit_button("Prediksi")

        

    if submit:

        input_data = pd.DataFrame([{
            'Origin_Facility': Origin_Facility,
            'Vehicle_Type': Vehicle_Type,
            'Route_Type': Route_Type,
            'Distance_KM': Distance_KM,
            'Package_Weight_KG': Package_Weight_KG,
            'Traffic_Conditions': traffic_map[Traffic_Conditions]
        }])
        
        input_data = pd.get_dummies(input_data, columns=['Origin_Facility', 'Vehicle_Type', 'Route_Type'])
        input_data = input_data.reindex(columns=x.columns, fill_value=0)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        st.success(f"Prediksi Emisi Karbon: {prediction[0]:.2f} kgCO₂e")
        

with tab2:
    st.header("Sebaran Data Jumlah Emisi Karbon")
    
    # Copy data
    df_cluster = df.copy()

    # Scaling
    cluster_scaler = MinMaxScaler()
    df_scaled = cluster_scaler.fit_transform(df_cluster)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(df_scaled)

    df_pca = pd.DataFrame(x_pca, columns=['PCA1', 'PCA2'])

    # Load model clustering
    kmeans = joblib.load("kmeans.pkl")

    # Prediksi cluster
    cluster_labels=kmeans.fit_predict(df_pca)
    df_pca['Cluster_Label'] = kmeans.labels_
    
    # Evaluasi
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(df_pca, kmeans.labels_)
    st.subheader("Evaluasi Cluster")
    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
    
    # Visualisasi
    st.subheader("Sebaran Data Emisi Karbon")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster_Label',palette='viridis', ax=ax)
    st.pyplot(fig)
    
    # Jumlah anggota cluster
    cluster_count = (df_pca['Cluster_Label'].value_counts().sort_index())
    st.subheader("Jumlah Anggota Tiap Cluster")
    st.dataframe(cluster_count)
    st.subheader(" ")
    
    fig, ax = plt.subplots()
    cluster_count.plot(kind='bar', ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")
    st.pyplot(fig)
    
    # Karakteristik cluster
    df_cluster['Cluster'] = df_pca['Cluster_Label']
    cluster_summary = df_cluster.groupby('Cluster')[
        [
            'Distance_KM',
            'Package_Weight_KG',
            'Carbon_Emission_kgCO2e'
        ]
    ].mean().round(2)

    st.subheader("Karakteristik Cluster")
    st.dataframe(cluster_summary)
    

