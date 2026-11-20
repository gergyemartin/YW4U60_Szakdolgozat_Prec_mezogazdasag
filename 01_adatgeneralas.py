"""

Budapesti Corvinus Egyetem
Gazdaságinformatikus BSc
Gergye Martin - SZAKDOLGOZAT
Precíziós mezőgazdaság üzemgazdasági megtérülésének vizsgálata szimulációs modellel
A szimulált üzemi adatállomány előállítása.



A szkript egy 24 táblás szántóföldi gazdaság 2019-2022 közötti adatait állítja elő.
A táblák egy része a vizsgált időszak alatt átáll precíziós technológiára.
Minden tábla több menedzsment zónára van osztva, és minden zónára minden évre
kiszámoljuk a hozamot, a költséget és a fedezetet.


Kimenet: szimulalt_adatok.xlsx
"""

import numpy as np
import pandas as pd

# =============================================================================
# 1. PARAMÉTEREK
# =============================================================================

# Random seed: Reprodukálhatóság miatt
MAG = 20260418
np.random.seed(MAG)

EVEK = [2019, 2020, 2021, 2022]
KULTURAK = ["őszi búza", "kukorica", "napraforgó", "őszi káposztarepce"]

# Országos átlaghozamok (t/ha) - KSH adatok
# A 2022-es aszály a tavaszi vetésűeket sújtotta, az őszi búzát kevésbé.
ORSZAGOS_HOZAM = {
    "őszi búza":          {2019: 5.30, 2020: 5.50, 2021: 6.00, 2022: 4.40},
    "kukorica":           {2019: 8.10, 2020: 8.60, 2021: 6.00, 2022: 3.40},
    "napraforgó":         {2019: 3.00, 2020: 2.80, 2021: 2.70, 2022: 1.80},
    "őszi káposztarepce": {2019: 3.00, 2020: 2.80, 2021: 2.80, 2022: 2.50},
}

# Terményárak (ezer Ft/t) - KSH felvásárlási árak
AR = {
    "őszi búza":          {2019:  51, 2020:  55, 2021:  72, 2022: 129},
    "kukorica":           {2019:  45, 2020:  49, 2021:  71, 2022: 110},
    "napraforgó":         {2019: 101, 2020: 121, 2021: 176, 2022: 269},
    "őszi káposztarepce": {2019: 116, 2020: 125, 2021: 174, 2022: 283},
}

# Inputköltségek bázisértékei (Ft/ha), 2019-es szinten
INPUT_BAZIS = {
    "őszi búza":          {"műtrágya": 60000, "vetőmag": 28000, "növényvédelem": 35000, "egyéb":  85000},
    "kukorica":           {"műtrágya": 85000, "vetőmag": 55000, "növényvédelem": 30000, "egyéb": 105000},
    "napraforgó":         {"műtrágya": 50000, "vetőmag": 42000, "növényvédelem": 38000, "egyéb":  90000},
    "őszi káposztarepce": {"műtrágya": 70000, "vetőmag": 32000, "növényvédelem": 48000, "egyéb":  90000},
}

# Inputárindexek: 2019 = 1,000 (pl.: A műtrágya 2022-re 3,7-szeresére drágult)
INPUT_INDEX = {
    "műtrágya":      {2019: 1.000, 2020: 0.921, 2021: 1.455, 2022: 3.725},
    "növényvédelem": {2019: 1.000, 2020: 1.010, 2021: 1.082, 2022: 1.320},
    "egyéb":         {2019: 1.000, 2020: 0.948, 2021: 1.119, 2022: 1.678},
    "vetőmag":       {2019: 1.000, 2020: 1.000, 2021: 1.100, 2022: 1.350},
}

TERULETALAPU_TAMOGATAS = 85000 # Ft/ha

# A precíziós technológia hatása az inputfelhasználásra
# 1,0 = nem változik (pl.: A műtrágya összmennyisége nem csökken, csak átrendeződik)
PREC_INPUT_SZORZO = {"műtrágya": 1.005, "vetőmag": 0.970,
                     "növényvédelem": 0.940, "egyéb": 1.000}
PREC_FOLYO_KOLTSEG = 4200 # Ft/ha: szoftver, adatelőkészítés, szaktanácsadás

# A precíziós kezelés hozamhatása három részből áll össze
HATAS_ALAP = 0.028                 # alaphatás homogén táblán, stresszmentes évben
HATAS_HETEROGENITAS = 0.38         # a tábla heterogenitásával arányos rész
HATAS_STRESSZ = 0.055              # az évjárati stresszel arányos rész

# Tanulási görbe: az átállás évében a hatás csak 45%-a érvényesül
FELFUTAS = {0: 0.45, 1: 0.75, 2: 1.00, 3: 1.00}

# A véletlen ingadozás három szintje (a hozam százalékában)
SZORAS_TABLA = 0.050               # tartós termőhelyi különbség
SZORAS_TABLA_EV = 0.055            # az adott tábla adott évi körülményei
SZORAS_MARADEK = 0.055             # zóna-év szintű zaj

N_TABLA = 24

# =============================================================================
# 2. ÉVJÁRATI STRESSZINDEX
# =============================================================================
# Minden kultúrára megnézzük, hogy az adott év hozama mennyivel marad el
# az adott kultúra legjobb évétől. Ez lesz a stresszindex (0 = legjobb év).

STRESSZ = {}
for kult in KULTURAK:
    legjobb = max(ORSZAGOS_HOZAM[kult].values())
    STRESSZ[kult] = {ev: round(1 - ORSZAGOS_HOZAM[kult][ev] / legjobb, 3) for ev in EVEK}

print("Évjárati stresszindex (0 = a kultúra legjobb éve):")
print(pd.DataFrame(STRESSZ).round(3))

# =============================================================================
# 3. TÁBLÁK LÉTREHOZÁSA
# =============================================================================
# Minden tábla kap méretet, zónaszámot, heterogenitást, átállási évet és egy tartós táblahatást (ez minden évben ugyanaz).

# Vetésforgó: tíz éves ciklus, a táblák eltérő pontról indulnak
FORGO = ["őszi búza", "őszi káposztarepce", "őszi búza", "kukorica",
         "napraforgó", "kukorica", "őszi búza", "kukorica",
         "napraforgó", "kukorica"]

tablak = []
for i in range(N_TABLA):
    # Átállás: 13 tábla áll át (8 db 2020-ban, 5 db 2021-ben), 11 nem áll át.
    # A lépcsőzetes átállás azért kell, hogy ugyanabban az évben legyen
    # kezelt és kezeletlen megfigyelés is. E nélkül a kezelés hatása nem választható el az évjárat hatásától.
    if i < 8:
        atallas = 2020
    elif i < 13:
        atallas = 2021
    else:
        atallas = None

    tablak.append({
        "tabla": f"T{i+1:02d}",
        "terulet_ha": round(np.random.uniform(21, 58), 1),
        "zonak_szama": np.random.randint(4, 7),
        "heterogenitas": round(np.random.uniform(0.03, 0.25), 3),
        "atallas_eve": atallas,
        "tabla_hatas": np.random.normal(0, SZORAS_TABLA),
        "forgo_kezdet": i % len(FORGO),
    })

tablak_df = pd.DataFrame(tablak)
print(f"\n{N_TABLA} tábla, összesen {tablak_df.terulet_ha.sum():.0f} ha")
print(f"Átálló táblák: {tablak_df.atallas_eve.notna().sum()} db")

# =============================================================================
# 4. AZ ADATSOROK ELŐÁLLÍTÁSA
# =============================================================================
# Három egymásba épülő ciklus: tábla -> év -> zóna
# Minden kör egy sort ad az adattáblához

# A tábla-év hatásokat előre kisorsoljuk
tabla_ev_hatas = {}
for t in tablak:
    for ev in EVEK:
        tabla_ev_hatas[(t["tabla"], ev)] = np.random.normal(0, SZORAS_TABLA_EV)

sorok = []

for t in tablak:
    for ev_index, ev in enumerate(EVEK):

        # Melyik kultúra van ezen a táblán ebben az évben?
        kultura = FORGO[(t["forgo_kezdet"] + ev_index) % len(FORGO)]

        # Precíziós-e a tábla ebben az évben?
        if t["atallas_eve"] is not None and ev >= t["atallas_eve"]:
            precizios = 1
            evek_atallas_ota = ev - t["atallas_eve"]
            felfutas = FELFUTAS[min(evek_atallas_ota, 3)]
        else:
            precizios = 0
            evek_atallas_ota = np.nan
            felfutas = 0.0

        # A kezelés hozamhatása ebben az évben, ezen a táblán
        hatas = felfutas * (HATAS_ALAP
                            + HATAS_HETEROGENITAS * t["heterogenitas"]
                            + HATAS_STRESSZ * STRESSZ[kultura][ev])

        # Zónaindexek: a tábla átlaga 1,0 körül, a szóródás a heterogenitás
        zona_indexek = np.random.normal(1.0, t["heterogenitas"], t["zonak_szama"])
        zona_indexek = zona_indexek / np.mean(zona_indexek)   # átlagtartó legyen

        for z in range(t["zonak_szama"]):
            zi = zona_indexek[z]

            # --- HOZAM ---
            zaj = np.random.normal(0, SZORAS_MARADEK)
            hozam = (ORSZAGOS_HOZAM[kultura][ev] * zi * (1 + hatas)
                     * np.exp(t["tabla_hatas"] + tabla_ev_hatas[(t["tabla"], ev)] + zaj))

            # --- INPUTKÖLTSÉGEK ---
            koltsegek = {}
            for tetel, bazis in INPUT_BAZIS[kultura].items():
                ertek = bazis * INPUT_INDEX[tetel][ev]
                # A zónaszintű differenciálás: a jobb zónák többet kapnak.
                # Csak a műtrágyát és a vetőmagot differenciáljuk.
                if precizios == 1 and tetel in ("műtrágya", "vetőmag"):
                    ertek = ertek * (1 + 0.55 * (zi - 1.0))
                if precizios == 1:
                    ertek = ertek * PREC_INPUT_SZORZO[tetel]
                koltsegek[tetel] = ertek

            kts_osszes = sum(koltsegek.values())
            if precizios == 1:
                kts_osszes += PREC_FOLYO_KOLTSEG

            # --- ÁRBEVÉTEL ÉS FEDEZET ---
            arbevetel = hozam * AR[kultura][ev] * 1000 + TERULETALAPU_TAMOGATAS
            fedezet = arbevetel - kts_osszes

            sorok.append({
                "tabla": t["tabla"],
                "ev": ev,
                "zona": z + 1,
                "kultura": kultura,
                "terulet_ha": t["terulet_ha"],
                "zona_index": round(zi, 4),
                "heterogenitas": t["heterogenitas"],
                "precizios": precizios,
                "evek_atallas_ota": evek_atallas_ota,
                "hozam_t_ha": round(hozam, 3),
                "mutragya_Ft_ha": round(koltsegek["műtrágya"]),
                "vetomag_Ft_ha": round(koltsegek["vetőmag"]),
                "novenyvedelem_Ft_ha": round(koltsegek["növényvédelem"]),
                "egyeb_Ft_ha": round(koltsegek["egyéb"]),
                "tech_koltseg_Ft_ha": PREC_FOLYO_KOLTSEG if precizios else 0,
                "koltseg_Ft_ha": round(kts_osszes),
                "arbevetel_Ft_ha": round(arbevetel),
                "fedezet_Ft_ha": round(fedezet),
            })

adatok = pd.DataFrame(sorok)

# Egy segédoszlop, amire a regresszióban szükség lesz: a kultúra és az év együtt, egyetlen kategóriaként.
# Erre azért van szükség, mert a 2022-es aszály nem egyformán érintette a kultúrákat és a kettő hatását együtt kell kezelni.
adatok["kultura_ev"] = adatok.kultura + " " + adatok.ev.astype(str)

print(f"\nAz állomány elkészült: {len(adatok)} sor, {adatok.shape[1]} oszlop")
print(f"Precíziós megfigyelések aránya: {adatok.precizios.mean():.1%}")

# =============================================================================
# 5. MENTÉS
# =============================================================================

adatok.to_excel("szimulalt_adatok.xlsx", index=False)
tablak_df.to_excel("tablak.xlsx", index=False)
print("\nMentve: szimulalt_adatok.xlsx, tablak.xlsx")

# Ellenőrzés: a hozamok évjárat szerinti alakulása
print("\nÁtlagos hozam kultúránként és évenként (t/ha):")
print(adatok.pivot_table(index="kultura", columns="ev",
                         values="hozam_t_ha", aggfunc="mean").round(2))
