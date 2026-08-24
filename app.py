# =====================================================================
#  app.py  |  Dashboard Bahaya Seismik Sesar Lembang (versi Streamlit)
# ---------------------------------------------------------------------
#  Andini Dwi Kurnia Putri - Statistika, Universitas Padjadjaran

# =====================================================================

import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium


# =====================================================================
# BAGIAN 0 | KONFIGURASI HALAMAN
# =====================================================================

st.set_page_config(
    page_title="Bahaya Seismik Sesar Lembang",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNA = {
    "utama":  "#1B5E4A",   # hijau tua - identitas
    "aksen":  "#C0392B",   # merah bata - garis fit, sesar
    "netral": "#5D6D7E",
    "terang": "#F4F6F7",
}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E5E8E8;
        border-left: 4px solid {WARNA['utama']};
        border-radius: 6px;
        padding: 12px 16px;
    }}
    .card-hijau {{
        background-color: {WARNA['utama']};
        color: white;
        padding: 8px 14px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        margin-bottom: 0px;
    }}
    .card-isi {{
        border: 1px solid #E5E8E8;
        border-top: none;
        border-radius: 0 0 6px 6px;
        padding: 14px;
        margin-bottom: 18px;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
        display: none;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        display: block;
        padding: 10px 14px;
        margin-bottom: 4px;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.15s;
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background-color: #E8F0ED;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        background-color: #1B5E4A;
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


def kartu_mulai(judul):
    """Header hijau ala bs4Dash. Dipanggil sebelum isi kartu."""
    st.markdown(f'<div class="card-hijau">{judul}</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-isi">', unsafe_allow_html=True)


def kartu_selesai():
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# BAGIAN 1 | KONFIGURASI SUMBER DATA
# =====================================================================
GH_USER   = "annkdr"
GH_REPO   = "sesar-lembang-hazard"
GH_BRANCH = "main"

BASE_URL = f"https://raw.githubusercontent.com/{GH_USER}/{GH_REPO}/{GH_BRANCH}/data/"


# =====================================================================
# BAGIAN 2 | PEMUATAN DATA
# =====================================================================

@st.cache_data(show_spinner=False)
def baca_csv(nama, wajib=True):
    try:
        return pd.read_csv(BASE_URL + nama)
    except Exception as e:
        if wajib:
            st.error(f"Gagal memuat {nama}: {e}")
        return None


@st.cache_data(show_spinner=False)
def baca_geojson(nama):
    import urllib.request
    try:
        with urllib.request.urlopen(BASE_URL + nama) as resp:
            return json.load(resp)
    except Exception as e:
        st.warning(f"Gagal memuat {nama}: {e}")
        return None


with st.spinner("Memuat data..."):
    katalog    = baca_csv("katalog_mainshock.csv")
    par_tab    = baca_csv("parameter_gr_hpp.csv")
    prob_tab   = baca_csv("tabel_probabilitas.csv",       wajib=False)
    sens_tab   = baca_csv("tabel_sensitivitas.csv")
    laju_tab   = baca_csv("tabel_laju_periode_ulang.csv", wajib=False)
    gft_tab    = baca_csv("tabel_gft.csv",                wajib=False)
    tembus_tab = baca_csv("tabel_tembus.csv",              wajib=False)
    trace_ll   = baca_csv("trace_lonlat.csv",             wajib=False)
    buffer_gj  = baca_geojson("buffer_25km.geojson")
    kab_gj     = baca_geojson("jabar_kabupaten_ringkas.geojson")

if katalog is None or par_tab is None:
    st.error(
        "Data inti tidak bisa dimuat. Periksa GH_USER di bagian atas "
        "app.py, dan pastikan repo GitHub Anda berstatus Public."
    )
    st.stop()


# --- Rapikan katalog ---
katalog.columns = [c.strip() for c in katalog.columns]
katalog["mag"] = pd.to_numeric(katalog["Magnitude"], errors="coerce")
katalog["dep"] = pd.to_numeric(katalog["Depth"], errors="coerce")
katalog["lat"] = pd.to_numeric(katalog["Latitude"], errors="coerce")
katalog["lon"] = pd.to_numeric(katalog["Longitude"], errors="coerce")
katalog["jrk"] = pd.to_numeric(katalog["jarak_sesar"], errors="coerce")
if "Date" in katalog.columns:
    if "Time (UTC)" in katalog.columns:
        gabung = katalog["Date"].astype(str) + " " + katalog["Time (UTC)"].astype(str)
        katalog["tanggal"] = pd.to_datetime(gabung, errors="coerce")
    else:
        katalog["tanggal"] = pd.to_datetime(katalog["Date"], errors="coerce")
    katalog["tahun"] = katalog["tanggal"].dt.year
else:
    katalog["tahun"] = np.nan


# --- Ambil parameter dari tabel ringkasan ---
# parameter_gr_hpp.csv berbentuk dua kolom (nama parameter, nilai).
# Fungsi ini mencari baris yang namanya cocok dengan pola teks, lalu
# mengambil nilainya .                                         
def ambil(pola, default=np.nan):
    col_nama = par_tab.columns[0]
    col_nilai = par_tab.columns[1]
    cocok = par_tab[col_nama].astype(str).str.contains(pola, case=False, regex=True)
    if not cocok.any():
        return default
    nilai = str(par_tab.loc[cocok, col_nilai].iloc[0]).replace(",", ".")
    try:
        return float(nilai)
    except ValueError:
        return default


BUFFER  = ambil(r"^Buffer",             25)
PANJANG = ambil(r"Panjang trace",       29.82)
N_TOT   = ambil(r"N mainshock",         89)
T_THN   = ambil(r"^T \(tahun|^T$",      17.57)
MC      = ambil(r"^Mc",                 2.7)
N_MC    = ambil(r"N di atas Mc",        52)
BVAL    = ambil(r"^b-value",            0.8951)
SDB     = ambil(r"^sd b",               0.1185)
BLO     = ambil(r"b CI bawah",          0.709)
BHI     = ambil(r"b CI atas",           1.164)
AVAL    = ambil(r"a-value tahunan",     2.8879)
MMAX    = ambil(r"Mmax",                7.0)
SLIP    = ambil(r"Slip rate",           4.5)

LAMBDA_MC = N_MC / T_THN
BETA      = BVAL * math.log(10)

MAG_MIN = float(np.floor(katalog["mag"].min() * 10) / 10)
MAG_MAX = float(np.ceil(katalog["mag"].max() * 10) / 10)
TAHUN_MIN = int(katalog["tahun"].min()) if katalog["tahun"].notna().any() else 2009
TAHUN_MAX = int(katalog["tahun"].max()) if katalog["tahun"].notna().any() else 2026


# =====================================================================
# BAGIAN 3 | FUNGSI INTI: LAJU DAN PROBABILITAS
# =====================================================================


def laju(m):
    """Laju tahunan kejadian M >= m, model GR terpotong di Mmax."""
    m = np.maximum(np.asarray(m, dtype=float), MC)
    atas  = np.exp(-BETA * (m - MC)) - np.exp(-BETA * (MMAX - MC))
    bawah = 1 - np.exp(-BETA * (MMAX - MC))
    hasil = LAMBDA_MC * atas / bawah
    return np.maximum(hasil, 0)


def periode_ulang(m):
    l = laju(m)
    with np.errstate(divide="ignore"):
        return np.where(l > 0, 1 / l, np.inf)


def prob_hpp(m, t):
    """Peluang minimal satu kejadian M>=m dalam t tahun (proses Poisson)."""
    return 1 - np.exp(-laju(m) * np.asarray(t, dtype=float))


def prob_tepat(k, m, t):
    """Peluang terjadi TEPAT k kejadian M>=m dalam t tahun."""
    lt = laju(m) * t
    return math.exp(-lt) * lt**k / math.factorial(k)


def fmt(x, d=2):
    """Format angka gaya Indonesia: koma sebagai desimal."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "\u221e"
    s = f"{x:,.{d}f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


# =====================================================================
# BAGIAN 4 | NAVIGASI
# =====================================================================
with st.sidebar:
    st.markdown(f"## 🌋 Sesar Lembang")
    halaman = st.radio(
        "Navigasi",
        ["Beranda", "Peta & Data", "Analisis GR", "Probabilitas", "Metodologi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("**Program Studi Statistika**  \nUniversitas Padjadjaran")
    st.caption("Data: BMKG, PuSGeN, BIG")


# =====================================================================
# HALAMAN 1 | BERANDA
# =====================================================================
if halaman == "Beranda":
    st.title("Bahaya Seismik Sesar Lembang")
    st.caption("Gutenberg-Richter | Homogeneous Poisson Process")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mainshock dianalisis", fmt(N_TOT, 0), f"radius {fmt(BUFFER,0)} km")
    c2.metric("Periode pengamatan", f"{fmt(T_THN,1)} th")
    c3.metric("b-value", fmt(BVAL, 3), f"sd {fmt(SDB,3)}")
    c4.metric("P(M\u22655) 50 th", f"{fmt(prob_hpp(5.0, 50)*100, 1)}%")

    st.write("")
    col_kiri, col_kanan = st.columns(2)

    with col_kiri:
        kartu_mulai("Alur analisis")
        st.markdown(f"""
1. **Deklusterisasi** — Gardner & Knopoff (1974) memisahkan gempa utama
   dari gempa susulan.
2. **Seleksi spasial** — hanya gempa dalam radius {fmt(BUFFER,0)} km dari
   trace sesar.
3. **Magnitude of completeness** — Mc = {fmt(MC,1)}, batas bawah katalog
   yang lengkap.
4. **Estimasi Gutenberg-Richter** — metode maximum likelihood (Aki-Utsu).
5. **Proyeksi probabilitas** — HPP dengan model GR terpotong di
   Mmax = {fmt(MMAX,1)}.
        """)
        kartu_selesai()

    with col_kanan:
        kartu_mulai("Model")
        st.markdown("**Gutenberg-Richter**")
        st.latex(r"\log_{10} N(M \geq m) = a - bm")
        st.caption(
            "Parameter b menyatakan proporsi gempa kecil terhadap gempa "
            "besar. Nilai b mendekati 1 lazim untuk sesar aktif."
        )
        st.markdown("**GR terpotong di Mmax**")
        st.latex(
            r"\lambda(m) = \lambda_{M_c}\,"
            r"\frac{e^{-\beta(m-M_c)} - e^{-\beta(M_{max}-M_c)}}"
            r"{1 - e^{-\beta(M_{max}-M_c)}}"
        )
        st.markdown("**Homogeneous Poisson Process**")
        st.latex(r"P(N \geq 1 \mid t) = 1 - e^{-\lambda t}")
        kartu_selesai()

    kartu_mulai("Parameter hasil estimasi")
    tabel_param = pd.DataFrame({
        "Parameter": ["Radius buffer", "Jumlah mainshock", "Periode pengamatan",
                      "Magnitude of completeness", "Jumlah kejadian di atas Mc",
                      "b-value", "Simpangan baku b", "a-value tahunan",
                      "Magnitudo maksimum", "Laju geser sesar"],
        "Nilai": [f"{fmt(BUFFER,0)} km", fmt(N_TOT,0), f"{fmt(T_THN,2)} tahun",
                  fmt(MC,1), fmt(N_MC,0), fmt(BVAL,4), fmt(SDB,4), fmt(AVAL,4),
                  fmt(MMAX,1), f"{fmt(SLIP,1)} mm/tahun"],
    })
    st.dataframe(tabel_param, hide_index=True, width="stretch")
    kartu_selesai()

    st.warning(
        "**Catatan.** Model ini mengasumsikan laju kejadian konstan sepanjang "
        "waktu (stasioner). Perbandingan dengan data paleoseismik perlu "
        "dilakukan secara terpisah, karena gempa karakteristik pada sesar "
        "tidak selalu mengikuti distribusi Gutenberg-Richter."
    )


# =====================================================================
# HALAMAN 2 | PETA & DATA
# =====================================================================
elif halaman == "Peta & Data":
    st.title("Peta & Data")

    with st.sidebar:
        st.markdown("### Kontrol tampilan")
        f_mag = st.slider("Rentang magnitudo", MAG_MIN, MAG_MAX,
                           (MAG_MIN, MAG_MAX), 0.1)
        f_tahun = st.slider("Rentang tahun", TAHUN_MIN, TAHUN_MAX,
                             (TAHUN_MIN, TAHUN_MAX))
        f_jarak = st.slider("Jarak maksimum ke sesar (km)", 0, int(math.ceil(BUFFER)),
                             int(math.ceil(BUFFER)))
        f_warna = st.radio("Warna titik menurut", ["Magnitudo", "Kedalaman"],
                            horizontal=True)
        f_layer = st.multiselect(
            "Tampilkan",
            ["Garis sesar", "Zona buffer", "Batas kabupaten"],
            default=["Garis sesar", "Zona buffer"],
        )
        st.caption("Ukuran lingkaran sebanding dengan magnitudo. "
                   "Klik titik untuk melihat rinciannya.")

    # --- Saring katalog ---
    d = katalog[
        (katalog["mag"] >= f_mag[0]) & (katalog["mag"] <= f_mag[1]) &
        (katalog["tahun"].isna() | katalog["tahun"].between(f_tahun[0], f_tahun[1])) &
        (katalog["jrk"].isna() | (katalog["jrk"] <= f_jarak))
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Kejadian", len(d))
    c2.metric("Magnitudo maksimum", f"M {d['mag'].max():.1f}" if len(d) else "-")
    c3.metric("Kedalaman median", f"{d['dep'].median():.0f} km" if len(d) else "-")

    kartu_mulai("Peta kejadian gempa utama")

    peta = folium.Map(location=[-6.84, 107.62], zoom_start=10, tiles="CartoDB positron")
    folium.TileLayer("OpenStreetMap", name="Jalan").add_to(peta)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satelit",
    ).add_to(peta)
    # Lapisan label (nama kota, jalan) - transparan, ditumpuk di atas
    # satelit supaya nama Bandung, Cimahi, Lembang, dst tetap terbaca.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Label kota & jalan",
        overlay=True, control=True, show=True,
    ).add_to(peta)

    if "Batas kabupaten" in f_layer and kab_gj is not None:
        folium.GeoJson(
            kab_gj, name="Batas kabupaten",
            style_function=lambda x: {"fillOpacity": 0, "color": "#95A5A6", "weight": 1},
        ).add_to(peta)

    if "Zona buffer" in f_layer and buffer_gj is not None:
        folium.GeoJson(
            buffer_gj, name="Zona buffer",
            style_function=lambda x: {
                "fillColor": WARNA["utama"], "fillOpacity": 0.08,
                "color": WARNA["utama"], "weight": 1.5, "dashArray": "5,5",
            },
        ).add_to(peta)

    if "Garis sesar" in f_layer and trace_ll is not None:
        garis = list(zip(trace_ll["lat"], trace_ll["lon"]))
        folium.PolyLine(garis, color=WARNA["aksen"], weight=4, opacity=0.9,
                        tooltip="Sesar Lembang").add_to(peta)

    if len(d):
        if f_warna == "Magnitudo":
            nilai = d["mag"]
            vmin, vmax = MAG_MIN, MAG_MAX
            judul_legenda = "Magnitudo"
        else:
            nilai = d["dep"]
            vmin, vmax = d["dep"].min(), d["dep"].max()
            judul_legenda = "Kedalaman (km)"

        import branca.colormap as cm
        pal = cm.LinearColormap(["#FFFFB2", "#FD8D3C", "#BD0026"], vmin=vmin, vmax=vmax)

        for _, r in d.iterrows():
            radius = max(3, (r["mag"] - MAG_MIN + 0.5) * 3)
            v = r["mag"] if f_warna == "Magnitudo" else r["dep"]
            popup = (f"<b>M {r['mag']}</b><br>"
                     f"Tanggal: {r.get('tanggal','-')}<br>"
                     f"Kedalaman: {r['dep']} km<br>"
                     f"Jarak ke sesar: {r['jrk']:.1f} km<br>"
                     f"Koordinat: {r['lat']:.3f}, {r['lon']:.3f}")
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=radius,
                color="white", weight=1, fill=True,
                fill_color=pal(v) if pd.notna(v) else "#999999",
                fill_opacity=0.75, popup=popup,
            ).add_to(peta)
        pal.caption = judul_legenda
        pal.add_to(peta)

    folium.LayerControl(collapsed=True).add_to(peta)
    st_folium(peta, width=None, height=550, returned_objects=[])
    kartu_selesai()

    kartu_mulai("Data terpilih")
    tampil = d.rename(columns={
        "tanggal": "Tanggal", "mag": "Magnitudo", "dep": "Kedalaman",
        "lat": "Lintang", "lon": "Bujur", "jrk": "Jarak ke sesar (km)",
    })[["Tanggal", "Magnitudo", "Kedalaman", "Lintang", "Bujur", "Jarak ke sesar (km)"]]
    st.dataframe(tampil, hide_index=True, width="stretch")
    st.download_button(
        "Unduh data (CSV)", tampil.to_csv(index=False).encode("utf-8"),
        "katalog_terfilter.csv", "text/csv",
    )
    kartu_selesai()


# =====================================================================
# HALAMAN 3 | ANALISIS GR
# =====================================================================
elif halaman == "Analisis GR":
    st.title("Gutenberg-Richter")

    col_kiri, col_kanan = st.columns([7, 5])

    with col_kiri:
        kartu_mulai("Sebaran frekuensi-magnitudo")

        m = np.sort(katalog["mag"].dropna().values)
        bin_ = np.arange(np.floor(m.min() * 10) / 10, MMAX + 1e-9, 0.1)
        kum = np.array([(m >= (x - 1e-9)).sum() for x in bin_])
        non = np.array([((np.abs(m - x)) < 0.05).sum() for x in bin_])

        grs_m = np.arange(MC, MMAX - 0.1, 0.05)
        grs_n = laju(grs_m) * T_THN

        fig = go.Figure()
        fig.add_scatter(x=bin_[kum > 0], y=kum[kum > 0], mode="markers",
                        name="Kumulatif",
                        marker=dict(color="#2c7fb8", size=7))
        fig.add_scatter(x=bin_[non > 0], y=non[non > 0], mode="markers",
                        name="Non-kumulatif",
                        marker=dict(color="#888888", size=6, symbol="triangle-up"))
        fig.add_scatter(x=grs_m, y=grs_n, mode="lines", name="Model GR terpotong",
                        line=dict(color=WARNA["aksen"], width=2.5))
        fig.add_shape(type="line", x0=MC, x1=MC, y0=0.8, y1=kum.max(),
                     line=dict(color=WARNA["utama"], dash="dash", width=1.5))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Magnitudo (Mw)", yaxis_title="Jumlah kejadian",
            yaxis_type="log",
            yaxis_range=[np.log10(0.05), np.log10(kum.max() * 2)],
            xaxis=dict(gridcolor="#EAECEE", zeroline=False),
            yaxis=dict(gridcolor="#EAECEE", zeroline=False),
            legend=dict(orientation="h", y=-0.2),
            height=430, margin=dict(t=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Titik biru: jumlah kumulatif gempa dengan magnitudo sama atau "
            "lebih besar. Segitiga: jumlah per selang. Garis putus hijau "
            "menandai Mc; fitting hanya memakai data di sebelah kanannya."
        )
        kartu_selesai()

    with col_kanan:
        kartu_mulai("Hasil pendugaan")
        st.markdown(f"**Nilai b**")
        st.markdown(f"## {fmt(BVAL, 4)}")
        st.caption(f"Galat baku {fmt(SDB,4)}. Selang kepercayaan bootstrap "
                   f"95 persen {fmt(BLO,3)} sampai {fmt(BHI,3)}.")
        st.markdown("**Nilai a tahunan**")
        st.markdown(f"## {fmt(AVAL, 4)}")
        st.caption(
            "Pendugaan memakai maximum likelihood (Aki, 1965; Utsu, 1965), "
            "bukan regresi kuadrat terkecil. Regresi memberi bobot sama pada "
            "tiap bin magnitudo, padahal bin gempa besar hanya berisi satu "
            "atau dua kejadian sehingga jauh lebih tidak pasti. Maximum "
            "likelihood memperhitungkan hal itu."
        )
        kartu_selesai()

    col_a, col_b = st.columns(2)

    with col_a:
        kartu_mulai("Penentuan magnitude of completeness")
        if gft_tab is not None:
            kol_mc = gft_tab.columns[gft_tab.columns.str.contains("mc", case=False)][0]
            kol_r  = gft_tab.columns[gft_tab.columns.str.contains(r"^r$|^r\b", case=False, regex=True)][0]
            puncak = gft_tab.loc[gft_tab[kol_r].idxmax(), kol_mc]

            fig2 = go.Figure()
            fig2.add_scatter(x=gft_tab[kol_mc], y=gft_tab[kol_r], mode="lines+markers",
                             line=dict(color=WARNA["utama"], width=2),
                             marker=dict(size=6, color=WARNA["utama"]))
            fig2.add_shape(type="line", x0=puncak, x1=puncak,
                          y0=0, y1=gft_tab[kol_r].max(),
                          line=dict(color=WARNA["aksen"], dash="dash"))
            fig2.update_layout(xaxis_title="Kandidat Mc",
                              yaxis_title="Goodness of fit R (%)",
                              showlegend=False, height=300, margin=dict(t=20))
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Data GFT tidak tersedia.")
        st.caption(
            "Goodness-of-fit test membandingkan sebaran amatan dengan sebaran "
            "sintetis pada berbagai nilai Mc. Nilai Mc terpilih adalah yang "
            "kecocokannya tertinggi."
        )
        kartu_selesai()

    with col_b:
        kartu_mulai("Uji sensitivitas radius buffer")
        kol_buf = sens_tab.columns[sens_tab.columns.str.contains("buffer", case=False)][0]
        kol_b_  = sens_tab.columns[sens_tab.columns.str.match("^b$", case=False)][0]
        kol_sd  = sens_tab.columns[sens_tab.columns.str.contains("sd", case=False)]
        sd = sens_tab[kol_sd[0]] if len(kol_sd) else 0

        fig3 = go.Figure()
        fig3.add_scatter(
            x=sens_tab[kol_buf], y=sens_tab[kol_b_], mode="lines+markers",
            name="b-value",
            error_y=dict(array=sd, color="#999999"),
            line=dict(color=WARNA["utama"], width=2),
            marker=dict(size=9, color=WARNA["utama"]),
        )
        fig3.add_shape(type="line", x0=BUFFER, x1=BUFFER,
                      y0=(sens_tab[kol_b_]-sd).min(), y1=(sens_tab[kol_b_]+sd).max(),
                      line=dict(color=WARNA["aksen"], dash="dash"))
        fig3.update_layout(xaxis_title="Radius buffer (km)", yaxis_title="b-value",
                          showlegend=False, height=300, margin=dict(t=20))
        st.plotly_chart(fig3, width="stretch")
        st.caption(
            "Nilai b relatif stabil pada radius 25 sampai 30 km lalu berubah "
            "tajam pada radius yang lebih besar. Perubahan itu menandakan "
            "masuknya kejadian dari sumber gempa lain."
        )
        kartu_selesai()

    col_c, col_d = st.columns(2)
    with col_c:
        kartu_mulai("Tabel goodness-of-fit")
        if gft_tab is not None:
            st.dataframe(gft_tab, hide_index=True, width="stretch", height=250)
        kartu_selesai()
    with col_d:
        kartu_mulai("Tabel sensitivitas")
        st.dataframe(sens_tab, hide_index=True, width="stretch", height=250)
        kartu_selesai()


# =====================================================================
# HALAMAN 4 | PROBABILITAS
# =====================================================================
elif halaman == "Probabilitas":
    st.title("Kalkulator Probabilitas")

    with st.sidebar:
        st.markdown("### Skenario")
        k_mag = st.slider("Magnitudo minimum (Mw)", round(MC, 1), MMAX, 5.0, 0.1)
        k_thn = st.slider("Horizon waktu (tahun)", 1, 200, 50, 1)
        st.caption(
            "Nilai dihitung langsung dari parameter hasil estimasi, bukan "
            "dibaca dari tabel. Setiap kombinasi magnitudo dan horizon "
            "menghasilkan angka baru."
        )
        st.caption(
            "**Membaca hasil:** probabilitas menyatakan peluang terjadi "
            "MINIMAL satu gempa dengan magnitudo sama atau lebih besar dari "
            "yang dipilih, dalam rentang waktu tersebut."
        )

    l = float(laju(k_mag))
    tr = float(periode_ulang(k_mag))
    p = float(prob_hpp(k_mag, k_thn))

    c1, c2, c3 = st.columns(3)
    c1.metric("Laju tahunan", fmt(l, 5), "kejadian per tahun")
    c2.metric("Periode ulang", f"{fmt(tr,1)} th" if not math.isinf(tr) else "\u221e")
    c3.metric("Probabilitas", f"{fmt(p*100,2)}%",
              f"M\u2265{fmt(k_mag,1)} dalam {k_thn} tahun")

    col_kiri, col_kanan = st.columns([7, 5])

    with col_kiri:
        kartu_mulai("Kurva probabilitas terhadap waktu")
        thn = np.arange(1, 201)
        mag_banding = [4.5, 5.0, 5.5, 6.0]
        warna_garis = ["#f0a202", "#b23a48", "#7b2cbf", "#2c7fb8"]

        fig = go.Figure()
        for mg, w in zip(mag_banding, warna_garis):
            fig.add_scatter(x=thn, y=prob_hpp(mg, thn) * 100, mode="lines",
                            name=f"M\u2265{fmt(mg,1)}",
                            line=dict(color=w, width=1.6, dash="dot"))
        fig.add_scatter(x=thn, y=prob_hpp(k_mag, thn) * 100, mode="lines",
                        name=f"Pilihan: M\u2265{fmt(k_mag,1)}",
                        line=dict(color=WARNA["utama"], width=4))
        fig.add_scatter(x=[k_thn], y=[p * 100], mode="markers", name="Skenario",
                        marker=dict(color=WARNA["utama"], size=13,
                                   line=dict(color="white", width=2)))
        fig.update_layout(
            xaxis_title="Horizon waktu (tahun)",
            yaxis_title="Probabilitas minimal 1 gempa (%)",
            yaxis_range=[0, 100], hovermode="x unified",
            legend=dict(orientation="h", y=-0.25), height=380, margin=dict(t=20),
        )
        st.plotly_chart(fig, width="stretch")
        kartu_selesai()

    with col_kanan:
        kartu_mulai("Distribusi jumlah kejadian")
        lt = l * k_thn
        kmax = min(12, max(4, int(np.ceil(lt + 3 * math.sqrt(max(lt, 1e-9))))))
        k_arr = np.arange(0, kmax + 1)
        p_arr = [prob_tepat(k, k_mag, k_thn) * 100 for k in k_arr]
        warna_bar = [WARNA["utama"] if k == math.floor(lt) else "#a8c5b4" for k in k_arr]

        fig2 = go.Figure()
        fig2.add_bar(x=k_arr, y=p_arr, marker_color=warna_bar)
        fig2.update_layout(
            xaxis_title="Jumlah gempa", yaxis_title="Probabilitas (%)",
            xaxis=dict(dtick=1), showlegend=False, height=380,
            margin=dict(t=30),
            title=dict(text=f"Rata-rata {fmt(lt,2)} kejadian dalam {k_thn} tahun",
                      font=dict(size=12, color="#666"), x=0.5),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Probabilitas terjadi tepat sekian gempa dalam horizon yang "
            "dipilih, mengikuti distribusi Poisson."
        )
        kartu_selesai()

    kartu_mulai("Tabel probabilitas lintas magnitudo")
    mags = np.arange(4.0, 6.5 + 1e-9, 0.5)
    tabel = pd.DataFrame({
        "Magnitudo": [fmt(x, 1) for x in mags],
        "Laju tahunan": [fmt(x, 5) for x in laju(mags)],
        "Periode ulang (th)": [fmt(x, 1) for x in periode_ulang(mags)],
    })
    for h in [10, 25, 50, 100]:
        tabel[f"P {h} th"] = [f"{fmt(x*100,2)}%" for x in prob_hpp(mags, h)]
    st.dataframe(tabel, hide_index=True, width="stretch")
    kartu_selesai()


# =====================================================================
# HALAMAN 5 | METODOLOGI
# =====================================================================
elif halaman == "Metodologi":
    st.title("Metodologi")

    col_kiri, col_kanan = st.columns([8, 4])

    with col_kiri:
        kartu_mulai("Catatan metodologi")
        st.markdown(f"""
##### Sumber data
Katalog gempa dari Badan Meteorologi, Klimatologi, dan Geofisika. Trace
dan nilai magnitudo maksimum sesar dari Peta Sumber dan Bahaya Gempa
Indonesia (PuSGeN, 2017).

##### Deklusterisasi
Metode Gardner & Knopoff (1974) dengan jendela ruang-waktu bergantung
magnitudo. Gempa susulan dihilangkan.

##### Pemilihan radius buffer
Radius {fmt(BUFFER,0)} km dipilih berdasarkan dua pertimbangan. Dari sisi
statistik, jumlah kejadian di atas Mc baru memadai untuk estimasi stabil
pada radius ini. Dari sisi geologi, radius yang lebih besar mulai
memasukkan gempa dari sumber lain, terlihat dari penurunan tajam nilai b.

##### Estimasi parameter
Nilai b diestimasi dengan maximum likelihood (Aki, 1965; Utsu, 1965),
bukan regresi kuadrat terkecil. MLE tidak memberi bobot berlebih pada
selang magnitudo besar yang hanya berisi satu-dua kejadian.

##### Keterbatasan
- Periode pengamatan {fmt(T_THN,1)} tahun relatif pendek dibandingkan
  periode ulang gempa besar, sehingga ekstrapolasi ke M besar memiliki
  ketidakpastian tinggi.
- Model Gutenberg-Richter mengasumsikan hubungan log-linear berlaku
  hingga Mmax. Sesar aktif sering menunjukkan perilaku gempa
  karakteristik yang menyimpang dari asumsi ini.
- Asumsi stasioneritas belum diuji formal. Metode Weichert (1980) dapat
  dipakai bila periode kelengkapan katalog berbeda antar rentang waktu.
        """)
        kartu_selesai()

    with col_kanan:
        kartu_mulai("Rujukan")
        st.markdown("""
<small>

- Aki, K. (1965). *Bulletin of the Earthquake Research Institute*, 43.
- Gardner, J. K., & Knopoff, L. (1974). *BSSA*, 64(5).
- Gutenberg, B., & Richter, C. F. (1944). *BSSA*, 34(4).
- Shi, Y., & Bolt, B. A. (1982). *BSSA*, 72(5).
- Wiemer, S., & Wyss, M. (2000). *BSSA*, 90(4).
- Weichert, D. H. (1980). *BSSA*, 70(4).
- Schwartz, D. P., & Coppersmith, K. J. (1984). *JGR*, 89(B7).
- Daryono, M. R., dkk. (2019). *Tectonophysics*.

</small>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"**Kode dan data:** [GitHub repository]"
                   f"(https://github.com/{GH_USER}/{GH_REPO})")
        kartu_selesai()
